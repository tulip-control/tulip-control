 /* ---------------------------------------------------------------------------*
 * FMU wrapper for TuLiP-control toolbox
 *
 * This file is based on the template FMU 'stepCounter' developed by
 * Christopher Brooks and Edward A. Lee and the template FMU 'stairsB'
 * developed by David Broman.  The latter can be found at
 *   trunk/ptolemy/actor/lib/fmi/fmus/stairsB/src/sources/stairsB.c
 * in the SVN repository
 *   https://repo.eecs.berkeley.edu/svn-anon/projects/eal/ptII
 * as of revision 69602.  The corresponding copyright statement and
 * license for that work immediately follow this comment block.
 *
 *
 * Authors: Yilin Mo
 * ---------------------------------------------------------------------------*/
/*
Below is the copyright agreement for the Ptolemy II system.
Version: $Id: copyright.txt 68472 2014-02-24 22:53:44Z cxh $

Copyright (c) 2013, 2014 The Regents of the University of California.
All rights reserved.

Permission is hereby granted, without written agreement and without
license or royalty fees, to use, copy, modify, and distribute this
software and its documentation for any purpose, provided that the above
copyright notice and the following two paragraphs appear in all copies
of this software.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
*/


#include <stdio.h>
#include <string.h>
#include "TuLiPFMU.h"

/* include fmu header files, typedefs and macros */
#include <fmiFunctions.h>
#include <TuLiPControl.h>

/* Used by FMI 2.0.  See FMIFuctions.h */
#define FMIAPI_FUNCTION_PREFIX TuLiPFMU_


/*****************************************************************************************
 * Data structure for an instance of this FMU.
 */
typedef struct {
	Controller *controller;
	pfloat* input;
	pfloat* output;
	idxint* dInput;
	fmiBoolean atBreakpoint;    /* Indicator that the first output at a step */
	/* time has been produced. */
	/* General states */
	fmiReal currentCount;       /* The current count (the output). */
	fmiReal lastSuccessfulTime; /* The time to which this FMU has advanced. */
	const fmiCallbackFunctions *functions;
	fmiString instanceName;
} ModelInstance;

/*****************************************************************************************
 *  Check various properties of this FMU. Return 0 if any requirement is violated, and 1 otherwise.
 *  @param instanceName The name of the instance.
 *  @param GUID The globally unique identifier for this FMU as understood by the master.
 *  @param modelGUID The globally unique identifier for this FMU as understood by this FMU.
 *  @param fmuResourceLocation A URI for the location of the unzipped FMU.
 *  @param functions The callback functions to allocate and free memory and log progress.
 *  @param visible Indicator of whether the FMU should run silently (fmiFalse) or interact
 *   with displays, etc. (fmiTrue) (ignored by this FMU).
 *  @param loggingOn Indicator of whether logging messages should be sent to the logger.
 *  @return The instance of this FMU, or null if there are required functions missing,
 *   if there is no instance name, or if the GUID does not match this FMU.
 */
int checkFMU(
		fmiString instanceName,
		fmiString GUID,
		fmiString modelGUID,
		fmiString fmuResourceLocation,
		const fmiCallbackFunctions *functions,
		fmiBoolean visible,
		fmiBoolean loggingOn)  {
	/* Logger callback is required. */
	if (!functions->logger) {
		return 0;
	}
	/* Functions to allocate and free memory are required. */
	if (!functions->allocateMemory || !functions->freeMemory) {
		functions->logger(NULL, instanceName, fmiError, "error",
				"fmiInstantiateSlave: Missing callback function: freeMemory");
		return 0;
	}
	if (!instanceName || strlen(instanceName)==0) {
		functions->logger(NULL, instanceName, fmiError, "error",
				"fmiInstantiateSlave: Missing instance name.");
		return 0;
	}
	if (strcmp(GUID, modelGUID)) {
		/* FIXME: Remove printfs. Replace with logger calls when they work. */
		fprintf(stderr,"fmiInstantiateSlave: Wrong GUID %s. Expected %s.\n", GUID, modelGUID);
		fflush(stderr);
		/*functions->logger(NULL, instanceName, fmiError, "error",
		  "fmiInstantiateSlave: Wrong GUID %s. Expected %s.", GUID, modelGUID); */
		return 0;
	}
	return 1;
}

/*****************************************************************************************
 *  Advance the state of this FMU from the current communication point to that point plus
 *  the specified step size.
 *  @param c The FMU.
 *  @param currentCommunicationPoint The time at the start of the step interval.
 *  @param communicationStepSize The width of the step interval.
 *  @param noSetFMUStatePriorToCurrentPoint True to assert that the master will not subsequently
 *   restore the state of this FMU or call fmiDoStep with a communication point less than the
 *   current one. An FMU may use this to determine that it is safe to take actions that have side
 *   effects, such as printing outputs. This FMU ignores this argument.
 *  @return fmiDiscard if the FMU rejects the step size, otherwise fmiOK.
 */
fmiStatus fmiDoStep(fmiComponent c, fmiReal currentCommunicationPoint,
		fmiReal communicationStepSize, fmiBoolean noSetFMUStatePriorToCurrentPoint) {
	ModelInstance* component = (ModelInstance *) c;

	/* If current time is greater than period * (value + 1), then it is
	   time for another increment. */
	double endOfStepTime = currentCommunicationPoint + communicationStepSize;
	double targetTime = TICK_PERIOD * (component->currentCount + 1);
	if (endOfStepTime >= targetTime - EPSILON) {
		/* It is time for an increment. */
		/* Is it too late for the increment? */
		if (endOfStepTime > targetTime + EPSILON) {
			/* Indicate that the last successful time is
			   at the target time. */
			component->lastSuccessfulTime = targetTime;
			fflush(stdout);
			return fmiDiscard;
		}
		/* We are at the target time. Are we
		   ready for the increment yet? Have to have already
		   completed one firing at this time. */
		if (component->atBreakpoint) {
			/* Not the first firing. Go ahead an increment. */
			component->currentCount++;

			input_function(component->controller,component->input,component->dInput);

			transition_function(component->controller);

			output_function(component->controller,component->output);
			/* Reset the indicator that the increment is needed. */
			component->atBreakpoint = fmiFalse;
		} else {
			/* This will complete the first firing at the target time.
			   We don't want to increment yet, but we set an indicator
			   that we have had a firing at this time. */
			fflush(stdout);
			component->atBreakpoint = fmiTrue;
		}
	}
	component->lastSuccessfulTime = endOfStepTime;
	fflush(stdout);
	return fmiOK;
}

/*****************************************************************************************
 *  Free memory allocated by this FMU instance.
 *  @param c The FMU.
 */
void fmiFreeSlaveInstance(fmiComponent c) {
	ModelInstance* component = (ModelInstance *) c;
	free_controller(component->controller);
	component->functions->freeMemory(component->input);
	component->functions->freeMemory(component->output);
	component->functions->freeMemory(component->dInput);
	component->functions->freeMemory(component);
}

/*****************************************************************************************
 *  Get the maximum next step size.
 *  If the last call to fmiDoStep() incremented the counter, then the maximum step
 *  size is zero. Otherwise, it is the time remaining until the next increment of the count.
 *  @param c The FMU.
 *  @param maxStepSize A pointer to a real into which to write the result.
 *  @return fmiOK.
 */
fmiStatus fmiGetMaxStepSize(fmiComponent c, fmiReal *maxStepSize) {
	ModelInstance* component = (ModelInstance *) c;
	if (component->atBreakpoint) {
		*maxStepSize = 0.0;
	} else {
		double targetTime = TICK_PERIOD * (component->currentCount + 1);
		double step = targetTime - component->lastSuccessfulTime;
		*maxStepSize = step;
	}
	return fmiOK;
}

/*****************************************************************************************
 *  Get the values of the specified real variables.
 *  @param c The FMU.
 *  @param vr An array of value references (indices) for the desired values.
 *  @param nvr The number of values desired (the length of vr).
 *  @param value The array into which to put the results.
 *  @return fmiError if a value reference is out of range, otherwise fmiOK.
 */
fmiStatus fmiGetReal(fmiComponent c, const fmiValueReference vr[], size_t nvr, fmiReal value[]) {
	int i, valueReference;
	ModelInstance* component = (ModelInstance *) c;

	for (i = 0; i < nvr; i++) {
		valueReference = vr[i];
		if (valueReference >= 0 && valueReference <p)
		{
			value[i] = (fmiReal) (*(component->output+valueReference));
		}
	}
	return fmiOK;
}

/*****************************************************************************************
 *  Get the specified FMU status. This procedure only provides status kind
 *  fmiLastSuccessfulTime. All other requests result in returning fmiDiscard.
 *  @param c The FMU.
 *  @param s The kind of status to return, which must be fmiLastSuccessfulTime.
 *  @param value A pointer to the location in which to deposit the status.
 *  @return fmiDiscard if the kind is not fmiLastSuccessfulTime, otherwise fmiOK.
 */
fmiStatus fmiGetRealStatus(fmiComponent c, const fmiStatusKind s, fmiReal* value) {
	ModelInstance* component = (ModelInstance *) c;
	if (s == fmiLastSuccessfulTime) {
		*value = component->lastSuccessfulTime;

		printf("fmiGetRealStatus returns lastSuccessfulTime is %g\n", *value);
		fflush(stdout);

		return fmiOK;
	}
	/* Since this FMU does not return fmiPending, there shouldn't be other queries of status. */
	return fmiDiscard;
}

/*****************************************************************************************
 *  Create an instance of this FMU.
 *  @param instanceName The name of the instance.
 *  @param GUID The globally unique identifier for this FMU.
 *  @param fmuResourceLocation A URI for the location of the unzipped FMU.
 *  @param functions The callback functions to allocate and free memory and log progress.
 *  @param visible Indicator of whether the FMU should run silently (fmiFalse) or interact
 *   with displays, etc. (fmiTrue) (ignored by this FMU).
 *  @param loggingOn Indicator of whether logging messages should be sent to the logger.
 *  @return The instance of this FMU, or null if there are required functions missing,
 *   if there is no instance name, or if the GUID does not match this FMU.
 */
fmiComponent fmiInstantiateSlave(
		fmiString instanceName,
		fmiString GUID,
		fmiString fmuResourceLocation,
		const fmiCallbackFunctions *functions,
		fmiBoolean visible,
		fmiBoolean loggingOn)  {
	ModelInstance* component;

	/* Perform checks. */
	if (!checkFMU(instanceName, GUID, MODEL_GUID, fmuResourceLocation, functions, visible, loggingOn)) {
		return NULL;
	}
	component = (ModelInstance *)functions->allocateMemory(1, sizeof(ModelInstance));
	component->currentCount = 0.0;
	component->lastSuccessfulTime = -1.0;
	component->atBreakpoint = fmiFalse;
	component->functions = functions;

	component->controller = instantiate_controller();
	component->input = (pfloat*)functions->allocateMemory(m,sizeof(pfloat));
	component->output= (pfloat*)functions->allocateMemory(p,sizeof(pfloat));
	component->dInput= (idxint*)functions->allocateMemory(nInputVariable,sizeof(idxint));

	/* Need to allocate memory and copy the string because JNA stores the string
	   in a temporary buffer that gets GC'd. */
	component->instanceName = (char*)functions->allocateMemory(1 + strlen(instanceName), sizeof(char));
	strcpy((char *)component->instanceName, instanceName);

	printf("%s: Invoked fmiInstantiateSlave.\n", component->instanceName);
	fflush(stdout);

	return component;
}

/*****************************************************************************************
 *  Initialize the FMU for co-simulation.
 *  @param c The FMU.
 *  @param relativeTolerance Suggested (local) tolerance in case the slave utilizes a
 *   numerical integrator with variable step size and error estimation (ignored by this FMU).
 *  @param tStart The start time (ignored by this FMU).
 *  @param stopTimeDefined fmiTrue to indicate that the stop time is defined (ignored by this FMU).
 *  @param tStop The stop time (ignored if stopTimeDefined is fmiFalse) (ignored by this FMU).
 *  @return fmiOK
 */
fmiStatus fmiInitializeSlave(fmiComponent c,
		fmiReal relativeTolerance,
		fmiReal tStart,
		fmiBoolean stopTimeDefined,
		fmiReal tStop) {

	ModelInstance* component = (ModelInstance *) c;
	printf("%s: Invoked fmiIntializeSlave: start: %g, StopTimeDefined: %d, tStop: %g..\n",
			component->instanceName, tStart, stopTimeDefined, tStop);
	fflush(stdout);

	component->lastSuccessfulTime = tStart;
	component->atBreakpoint = fmiFalse;

	init_controller(component->controller);

	printf("successful init controller\n");
	fflush(stdout);
	output_function(component->controller,component->output);

	return fmiOK;
}

/*****************************************************************************************
 *  Set the specified real values.
 *  @param c The FMU.
 *  @param vr An array of indexes of the real variables to be set (value references).
 *  @param nvr The number of values to be set (the length of the array vr).
 *  @param value The values to assign to these variables.
 *  @return fmiError if a value reference is out of range, otherwise fmiOK.
 */
fmiStatus fmiSetReal(fmiComponent c, const fmiValueReference vr[], size_t nvr, const fmiReal value[]){
	int i, valueReference;
	ModelInstance* component = (ModelInstance *) c;
	for (i = 0; i < nvr; i++) {
		valueReference = vr[i];
		if (valueReference >= p && valueReference < p+m)
		{
			*(component->input+valueReference-p)=(pfloat) value[i];
		}
	}
	return fmiOK;
}

/*****************************************************************************************
 *  Set the specified integer values.
 *  @param c The FMU.
 *  @param vr An array of indexes of the integer variables to be set (value references).
 *  @param nvr The number of values to be set (the length of the array vr).
 *  @param value The values to assign to these variables.
 *  @return fmiError if a value reference is out of range, otherwise fmiOK.
 */
fmiStatus fmiSetInteger(fmiComponent c, const fmiValueReference vr[], size_t nvr, const fmiInteger value[]){
	int i, valueReference;
	ModelInstance* component = (ModelInstance *) c;
	for (i = 0; i < nvr; i++) {
		valueReference = vr[i];
		if (valueReference >= p+m && valueReference < p+m+nInputVariable)
		{
			*(component->dInput+valueReference-p-m)=(idxint) value[i];
		}
	}
	return fmiOK;
}

/*****************************************************************************************
 *  Terminate this FMU. This does nothing, since this FMU is passive.
 *  @param c The FMU.
 *  @return fmiOK if the FMU was non-null, otherwise return fmiError
 */
fmiStatus fmiTerminateSlave(fmiComponent c) {
	ModelInstance* component = (ModelInstance *) c;

	if (component == NULL) {
		printf("fmiTerminateSlave called with a null argument?  This can happen while exiting during a failure to construct the component\n");
		fflush(stdout);
		return fmiError;
	} else {
		printf("%s: fmiTerminateSlave\n", component->instanceName);
		fflush(stdout);
	}

	return fmiOK;
}

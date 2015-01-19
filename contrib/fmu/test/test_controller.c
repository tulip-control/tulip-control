#include "TuLiPControl.h"

int main()
{
	int i,j,t;
	Controller *c;
	idxint *dInput = malloc(sizeof(idxint)*nInputVariable);
	pfloat *x = malloc(sizeof(pfloat)*n);
	pfloat *nextx = malloc(sizeof(pfloat)*n);
	pfloat *u = malloc(sizeof(pfloat)*p);
	for(i=0;i<n;i++)
	{
		*(x+i) = x0[i];
	}

	c = instantiate_controller();
	init_controller(c);

	printf("This is the robot_planning/continuous.py example\n");
	printf("The robot should visit home [0,1]*[0,1] infinitely often\n");
	printf("The robot should visit lot [2,3]*[1,2] after the parking signal is set to 1\n");

	t = 0;
	printf("Time: %d, State: ", t);
	for(i=0;i<n;i++)
	{
		printf("%5.3f, ", *(x+i));
	}
	printf("\n");

	while(1)
	{
		output_function(c, u);
		/* reset nextx */
		for(i=0;i<n;i++)
		{
			*(nextx+i) = 0;
		}
		/* compute Ax(k) first */
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{
				*(nextx+i) += (*(A+i+j*n)) * (*(x+j));
			}
		}
		/* add Bu(k) */
		for(i=0;i<n;i++)
		{
			for(j=0;j<p;j++)
			{
				*(nextx+i) += (*(B+i+j*n)) * (*(u+j));
			}
		}
		/* copy nextx to x */
		for(i=0;i<n;i++)
		{
			*(x+i) = *(nextx+i);
		}

		t++;
		printf("Time: %d, State: ", t);
		for(i=0;i<n;i++)
		{
			printf("%5.3f, ", *(x+i));
		}
		printf("\n");

		if (t%totalSteps == 0)
		{
			printf("Enter the new discrete input:\n");
			for(i=0;i<nInputVariable;i++)
			{
				printf("input -%2d, range [0 -%2d]: ", i, nInputValue[i]-1);
				scanf("%d", dInput+i);
			}
		}
		input_function(c, x, dInput);
		transition_function(c);
		if (t%totalSteps == 0)
		{
			printf("\nState of the Mealy machine: %d",c->fsm->currentState);
			printf("\nDiscrete Region: %d",c->dRegion);
			printf("\nNext goal: %d\n\n",c->goal);
		}
	}
	free_controller(c);
	free(x);
	free(nextx);
	free(u);
	free(dInput);
	return 0;
}

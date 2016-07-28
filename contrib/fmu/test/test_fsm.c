#include <stdio.h>
#include "FSM.h"

int main(void)
{
	FSM* fsm = init_fsm();
	int value0[] = {0};
	int value1[] = {1};
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value0));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value1));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value1));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value0));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value0));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value0));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value1));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value1));
	printf("(%d,%d),",fsm->currentState,fsm_transition(fsm,value0));
	free_fsm(fsm);
	return 0;
}

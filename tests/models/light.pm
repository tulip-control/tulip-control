dtmc

module light

	// state
	s : [0..1] init 0;

	[] s=0 -> 0.2 : (s'=1) + 0.8 : (s'=0);
	[] s=1 -> 0.5 : (s'=0) + 0.5 : (s'=1);

endmodule

label "green" = s=0;
label "red" = s=1;

dtmc

module mh

	// state
	s : [0..6] init 0;

	[] s=0 -> 0.8 : (s'=1) + 0.2 : (s'=0);
	[] s=1 -> 0.8 : (s'=2) + 0.2 : (s'=1);
	[] s=2 -> 0.8 : (s'=3) + 0.2 : (s'=2);
	[] s=3 -> 0.8 : (s'=4) + 0.2 : (s'=3);
	[] s=4 -> 0.8 : (s'=5) + 0.2 : (s'=4);
	[] s=5 -> 0.8 : (s'=6) + 0.2 : (s'=5);
	[] s=6 -> 1: (s'=6);

endmodule

label "h0" = s=0;
label "h1" = s=1;
label "h2" = s=2;
label "h3" = s=3;
label "h4" = s=4;
label "h5" = s=5;
label "h6" = s=6;

all:
	sudo python setup.py install nocheck --record files.txt 
install:
	sudo python setup.py install nocheck --record files.txt 
develop:
	sudo python setup.py develop nocheck --record files.txt 
clean: 
	sudo cat files.txt | xargs rm -rf
	
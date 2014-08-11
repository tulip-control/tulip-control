all:
	python setup.py install nocheck --record files.txt 

clean: 
	sudo cat files.txt | xargs rm -rf
	
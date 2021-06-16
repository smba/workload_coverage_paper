filename=main

pdf: ps
	ps2pdf ${filename}.ps

ps:	dvi
	dvips -t letter ${filename}.dvi

dvi:
	latex ${filename}
	bibtex ${filename}||true
	latex ${filename}
	latex ${filename}

clean:
	rm -f ${filename}.{ps,pdf,log,aux,out,dvi,bbl,blg}

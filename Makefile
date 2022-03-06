TARGET=build/
AUX=aux/
DOCUMENT=main

pdf:	clean
		mkdir -p ${TARGET}
		mkdir -p ${AUX}
		pdflatex  -aux-directory=${AUX} -output-directory=${TARGET} ${DOCUMENT}.tex
		bibtex ${TARGET}${DOCUMENT} > /dev/null
		pdflatex  -aux-directory=${AUX} -output-directory=${TARGET} ${DOCUMENT}.tex
		pdflatex  -aux-directory=${AUX} -output-directory=${TARGET} ${DOCUMENT}.tex

clean:
	rm -f ${TARGET}*.log
	rm -f ${TARGET}*.aux
	rm -f ${TARGET}*.bbl
	rm -f ${TARGET}*.bcf
	rm -f ${TARGET}*.blg
	rm -f ${TARGET}*.out
	rm -f ${TARGET}*.toc
	rm -f ${TARGET}*.lof
	rm -f ${TARGET}*.lot
	rm -f ${TARGET}*.dvi
	rm -f ${TARGET}*.xml
	rm -rf ${TARGET}

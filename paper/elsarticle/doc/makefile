# $Id: makefile 282 2026-01-08 05:54:22Z rishi $

file=elsdoc


all: pdf out 
	make pdf
	make pdf

#out:
#	if  [ -f $(file).out ] ; then cp $(file).out tmp.out; fi ;
#	sed 's/BOOKMARK/dtxmark/g;' tmp.out > x.out; mv x.out tmp.out ;

out:
	if  [ -f "$(file).out" ] ; then cp "$(file).out" tmp.out; fi ; 
	sed 's/BOOKMARK/dtxmark/g;' tmp.out > x.out; mv x.out tmp.out ; 
	sed -i 's/{\\376\\377/{/g' tmp.out ; 
	sed -i '/dtxmark/s/\\000//g' tmp.out
	sed -i 's/\\040/ /g' tmp.out ; 
	sed -i 's/\t/ /g' tmp.out

pdf:
	pdflatex $(file).tex

index:
	makeindex -s gind.ist -o $(file).ind $(file).idx 

changes:
	makeindex -s gglo.ist -o $(file).gls $(file).glo

xview:
#	xpdf -z 200 $(file).pdf &>/dev/null
	open -a 'Skim.app' $(file).pdf 

view:
	open -a 'Adobe Reader.app' $(file).pdf

ins:
	latex $(file).ins 

diff:
	diff $(file).sty ../$(file).sty |less

copy:
	cp $(file).sty ../


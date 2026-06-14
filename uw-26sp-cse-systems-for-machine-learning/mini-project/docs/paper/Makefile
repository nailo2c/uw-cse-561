.PHONY: pdf clean

pdf:
	pandoc paper.md \
		--metadata-file=paper-metadata.yaml \
		--pdf-engine=pdflatex \
		--resource-path=../.. \
		-o paper.pdf

clean:
	rm -f paper.pdf

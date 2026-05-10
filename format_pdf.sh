#!/bin/bash
# pdfseparate public/relatorio.pdf public/page-%d.pdf
# num_pages=$(pdfinfo public/relatorio.pdf | grep "Pages" | awk '{print $2}')
# pdfunite public/page-{2..$num_pages}.pdf public/relatorio_final.pdf
# sleep 1
# rm public/page-*.pdf

qpdf public/relatorio.pdf --pages . 2-z -- public/relatorio_final.pdf
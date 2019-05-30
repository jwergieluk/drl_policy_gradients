set term pdfcairo enhanced font ",12"
set output "scores.pdf"

set style line 1	lt 1	pt 12	ps 2	lw 4	lc rgb	'#DCDC00'
set style line 2	lt 1	pt 12	ps 2	lw 4	lc rgb	'#8E388E'
set style line 3	lt 1	pt 12	ps 2	lw 4	lc rgb	'#E30000'
set style line 4	lt 1	pt 12	ps 2	lw 4	lc rgb	'#CD6600'
set style line 5 	lt 1 	pt 12	ps 2	lw 4	lc rgb  '#228B8B'
set style line 6 	lt 1 	pt 12	ps 2	lw 4	lc rgb  '#D8C8D8'

set grid
set datafile separator ','
set key right bot

set xlabel "Episode"
set ylabel "Reward"

plot "scores.csv" u 1:2 w l ls 5 t "reward", \
"scores.csv"  u 1:3 w l ls 2 t "average reward (100 ep.)"

# vim: set ft=gnuplot:

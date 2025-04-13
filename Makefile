SRCDIR = src

DATASET = datasets/dataset_train.csv

PKGS = matplotlib pandas numpy seaborn scikit-learn

.SILENT:

all: check
	printf "\x1B[32m Packages Ready!\x1B[0m\n"
	printf "\x1B[30m  Usage:\x1B[33m make {[d], [h,s,p], [t,e]}\x1B[0m\n"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg; \
		fi; \
	done

d:
	python3 $(SRCDIR)/describe.py $(DATASET)

h:
	python3 $(SRCDIR)/visualize/histogram.py $(DATASET)

s:
	python3 $(SRCDIR)/visualize/scatter_plot.py $(DATASET)

p:
	python3 $(SRCDIR)/visualize/pair_plot.py $(DATASET)

b:
	python3 $(SRCDIR)/visualize/box_plot.py $(DATASET)

t:
	python3 $(SRCDIR)/model/logreg_train.py $(DATASET)

e:
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv thetas.csv

t1:
	python3 $(SRCDIR)/model/logreg_train.py datasets/ex_train.csv

e1:
	python3 $(SRCDIR)/model/logreg_predict.py datasets/ex_test.csv thetas.csv

a:
	python3 $(SRCDIR)/model/ada_train.py datasets/ex_train.csv

gen:
	python3 datasets/split.py $(DATASET)

clean:
	rm -rf houses.csv

fclean: clean
	rm -rf thetas.csv

gpush: fclean
	git add .
	git commit -m "optimize"
	git push

re: fclean all

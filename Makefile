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
			pip3 install $$pkg > /dev/null 2>&1; \
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
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv weights.csv

a: t e

o:
	python3 $(SRCDIR)/model/old_logreg_train.py $(DATASET)
	

bonus:
	python3 $(SRCDIR)/model/ada_train.py datasets/ex_train.csv

gen:
	python3 datasets/split.py $(DATASET)

clean:
	rm -rf houses.csv
	find . -name .DS_Store -delete

fclean: clean
	rm -rf weights.csv

gpush: fclean
	git add .
	git commit -m "Rajs Function"
	git push

re: fclean all

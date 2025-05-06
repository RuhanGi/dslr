SRCDIR = src

DATASET = datasets/dataset_train.csv

PKGS = matplotlib pandas numpy seaborn scikit-learn

.SILENT:

all: check
	printf "\x1B[32m Packages Ready!\x1B[0m\n"
	printf "\x1B[30m  Usage:\x1B[33m make {[d], [h,s,p,b], [t,e,a]}\x1B[0m\n"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg > /dev/null 2>&1; \
		fi; \
	done

d:
	python3 $(SRCDIR)/analysis/describe.py $(DATASET)

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

a:
	python3 $(SRCDIR)/model/ada_train.py $(DATASET)
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv batch.csv
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv stochastic.csv
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv minibatch.csv
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv adagrad.csv
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv rmsprop.csv
	python3 $(SRCDIR)/model/logreg_predict.py datasets/dataset_test.csv adam.csv

bonus:
	python3 $(SRCDIR)/model/ada_train.py datasets/ex_train.csv

clean:
	rm -rf houses.csv
	find . -name .DS_Store -delete

fclean: clean
	rm -rf weights.csv batch.csv stochastic.csv minibatch.csv adagrad.csv rmsprop.csv adam.csv

gpush: fclean
	git add .
	git commit -m "Optimized"
	git push

re: fclean all

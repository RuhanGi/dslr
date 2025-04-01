SRCDIR = src

# DATASET = datasets/small.csv
DATASET = datasets/dataset_train.csv

PKGS = matplotlib pandas numpy seaborn

.SILENT:

all: check
	# printf "\x1B[32m Model Trained!\x1B[0m\n"

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

clean:
	# rm -rf abc.txt

fclean: clean
	# rm -rf abc.txt

gpush: fclean
	git add .
	git commit -m "first"
	git push

re: fclean all

GRAPHDIR = $(HOME)/laplacians
# GRAPHS=$(file < graphs.txt)
GRAPHS=$(shell cat graphs.txt)
GRAPHS_MTX_1=$(GRAPHS:%=$(GRAPHDIR)/%.mtx)
GRAPHS_MTX_2=$(wildcard $(GRAPHDIR)/*.mtx)
GRAPHS_MTX=$(filter $(GRAPHS_MTX_1),$(GRAPHS_MTX_2)) # intersection
ALL_GRAPHS=$(GRAPHS_MTX:$(GRAPHDIR)/%.mtx=%)

all: ligmg.csv

.PHONY: clean

clean:
	$(RM) $(LIGMG_GRAPHS) $(LAMG_GRAPHS)

%.ligmg.csv: $(GRAPHDIR)/%.mtx
	./ligmg $< $(LIGMG_FLAGS) --csv $@

ligmg.csv: $(ALL_GRAPHS:%=%.ligmg.csv)
	../../scripts/join_csv.bash $^ > $@

%.metric.csv: $(GRAPHDIR)/%.mtx
	julia ../../scripts/graph_metrics.jl $< > $@

metric.csv: $(ALL_GRAPHS:%=%.metric.csv)
	../../scripts/join_csv.bash $^ > $@

.SECONDARY: $(ALL_GRAPHS:%=%.eig.small.h5)
.SECONDARY: $(ALL_GRAPHS:%=%.eig.large.h5)

all_eig: $(ALL_GRAPHS:%=%.eig.small.h5) $(ALL_GRAPHS:%=%.eig.large.h5)

%.eig.small.h5: $(GRAPHDIR)/%.mtx
	./ligmg_eigensolve $< $@ -eps_monitor -eps_smallest_magnitude -st_pc_type ligmg -st_ksp_type preonly -st_type sinvert -eps_target 0 -eps_nev 10

%.eig.large.h5: $(GRAPHDIR)/%.mtx
	./ligmg_eigensolve $< $@ -eps_monitor -eps_largest_magnitude -eps_nev 10

%.vec: %.eig.small.h5
	julia ../../util/eigvals_to_vec.jl $< > $@

%.hard.ligmg.csv: $(GRAPHDIR)/%.mtx %.vec
	./ligmg $< --rhs $(word 2,$^) --csv $@

hard.ligmg.csv: $(ALL_GRAPHS:%=%.hard.ligmg.csv)
	../../scripts/join_csv.bash $^ > $@

%.pcg.csv: $(GRAPHDIR)/%.mtx
	./ligmg $< --test-cg --no-test-ligmg --csv $@

pcg.csv: $(ALL_GRAPHS:%=%.pcg.csv)
	../../scripts/join_csv.bash $^ > $@

%.hard.pcg.csv: $(GRAPHDIR)/%.mtx %.vec
	./ligmg $< --rhs $(word 2,$^) --test-cg --no-test-ligmg --csv $@

hard.pcg.csv: $(ALL_GRAPHS:%=%.hard.pcg.csv)
	../../scripts/join_csv.bash $^ > $@

# %.lamg.csv: $(GRAPHDIR)/%.mat
# 	timeout 30m julia ../scripts/compare_solvers.jl $< $(LAMG_FLAGS) --dump $@

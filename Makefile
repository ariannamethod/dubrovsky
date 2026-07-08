# Dubrovsky Makefile — Go binary (notorch sgemv) + reference C build.
#
# Default target builds the Go CLI which is what you actually want.
# `make alexey` builds the legacy zero-deps C reference (kept for posterity
# and for environments without cgo).

.PHONY: dubrovsky alexey run clean test

# Go binary: REPL + one-shot, cgo into vendored notorch.
dubrovsky:
	go build -o dubrovsky ./cmd/dubrovsky/

# Reference C build (no cgo, no BLAS — slower, but trivially portable).
alexey:
	cc -O3 -march=native -o alexey dubrovsky.c -lm

# Build + drop into the REPL.
run: dubrovsky
	./dubrovsky

clean:
	rm -f dubrovsky alexey

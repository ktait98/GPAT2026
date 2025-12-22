all: boxm

#Find the nf-config executable from netcdf-fortran that gives us the netcdf setup
NFCONFIG=$(which nf-config)

#Pick a compiler and flags that will work with our netcdf setup, hopefully
FCOMP=$(shell nf-config --fc)
FFLAG=$(shell nf-config --fflags)
FLIBS=$(shell nf-config --flibs)
CLIBS=$(shell nc-config --libs)

#Annoyingly the above does not provide a LD_LIBRARY_PATH type result, so to run things
#one will still need to add them, this should do:

CLPATH=$(shell nc-config --libdir)
FLPATH=$(shell nf-config --prefix)

# We have to guess that nf-config libs are in /lib and not /lib64, nf-config has no --libdir
LPATH=${CLPATH}:${FLPATH}/lib

RUNDIR=/home/ktait98/GPAT2025/pycontrails_kt/pycontrails/
DATADIR=/home/ktait98/GPAT2025/pycontrails_kt/pycontrails/models/gpat/data/

#Anything relying on PHONY will ALWAYS be rebuilt when make us run
PHONY:

# Directories to be created
DIRS = $(DATADIR)inputs $(DATADIR)outputs $(DATADIR)inputs/glob

# Target to change permissions of the directory
.PHONY: chmod_dirs
chmod_dirs: create_dirs
	chmod 755 $(DATADIR)

# Target to create directories
.PHONY: create_dirs
create_dirs:
	mkdir -p $(DIRS)

# Target to download files from Google Drive
.PHONY: download_files
download_files: create_dirs
	@if [ ! -f $(DATADIR)inputs/glob/species.nc ]; then \
		./download_script.sh 'https://drive.google.com/uc?export=download&id=1s0RtRh427dzxe4K0Y4_tsG-1r-nlDV57' $(DATADIR)inputs/glob/species.nc; \
	fi
	@if [ ! -f $(DATADIR)inputs/glob/h2o_concs.nc ]; then \
		wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1FPCxXA9dHgSkHnK8kP6RD0R94OpfrKJW' -O $(DATADIR)inputs/glob/h2o_concs.nc; \
	fi
	@if [ ! -f $(DATADIR)inputs/glob/air_temperature.nc ]; then \
		wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1KXazdpDcbzmsoxd5JUjMc40czf2Wd992' -O $(DATADIR)inputs/glob/air_temperature.nc; \
	fi

# Main target to build the boxm executable
boxm: create_dirs download_files boxm.f90 Makefile
	${FCOMP} -cpp -o boxm boxm.f90 ${FFLAG} ${FLIBS} ${CLIBS} -DPYCONTRAILSDIR=${RUNDIR}

# boxm_clean: create_dirs download_files boxm_clean.f90 Makefile
# 	${FCOMP} -cpp -o boxm_clean boxm_clean.f90 ${FFLAG} ${FLIBS} ${CLIBS} -DPYCONTRAILSDIR=${RUNDIR}

# boxm_orig: boxm_orig.for Makefile
# 	${FCOMP} -o boxm_orig boxm_orig.for ${FFLAG} ${FLIBS} ${CLIBS}

# Clean target to remove the executable and directories
.PHONY: clean
clean:
	rm -f boxm
	rm -rf $(DIRS)

test: boxm
	LD_LIBRARY_PATH=${LPATH}:${LD_LIBRARY_PATH} ./boxm
FROM nixos/nix 

# setup nix
RUN nix-env -iA cachix -f https://cachix.org/api/v1/install
RUN cachix use vulcanclimatemodeling
RUN nix-env -iA nixpkgs.git nixpkgs.openssh
COPY shell.nix /fv3gfs-wrapper/
SHELL ["nix-shell", "/fv3gfs-wrapper/shell.nix", "--run"]

# run the nix-shell once for caching
RUN echo "caching"


# Copy inputs required for building the werapper
COPY fv3gfs/wrapper/dynamics_properties.json fv3gfs/wrapper/physics_properties.json /fv3gfs-wrapper/fv3gfs/wrapper/
COPY fill_templates.py HISTORY.md LICENSE Makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-wrapper/
COPY fv3gfs /fv3gfs-wrapper/fv3gfs
COPY templates /fv3gfs-wrapper/templates
COPY lib/ /fv3gfs-wrapper/lib/
RUN cd /fv3gfs-wrapper && make -C lib clean && make -C lib && python setup.py develop

# Copy things needed for the docs
COPY examples /fv3gfs-wrapper/examples
COPY docs /fv3gfs-wrapper/docs


ENV FV3CONFIG_CACHE_DIR=/fv3config-cache MPI=mpich

RUN mkdir $FV3CONFIG_CACHE_DIR && \
    chmod -R 777 $FV3CONFIG_CACHE_DIR && \
    echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# copy the tests
COPY tests /fv3gfs-wrapper/tests

# interactive shell by default
CMD ["bash"]

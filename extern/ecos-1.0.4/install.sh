# install ecos-1.0.4
wget https://github.com/ifa-ethz/ecos/archive/v1.0.4.tar.gz
tar xzf v1.0.4.tar.gz --strip 1
sed -i '10s/-Wextra/-Wextra -fPIC/' ecos.mk
make ecos


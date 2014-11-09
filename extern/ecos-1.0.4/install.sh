# install ecos-1.0.4
wget https://github.com/ifa-ethz/ecos/archive/v1.0.4.tar.gz
MD5CHECKSUM=9846bfa7817d3669f3c9269bf5567811
FILECHECKSUM=`md5sum v1.0.4.tar.gz| sed 's/ .*//'`
if [ "$MD5CHECKSUM" = "$FILECHECKSUM" ]
then
	tar xzf v1.0.4.tar.gz --strip 1
	sed -i '10s/-Wextra/-Wextra -fPIC/' ecos.mk
	make ecos
else
	echo "The checksums do not match"
fi


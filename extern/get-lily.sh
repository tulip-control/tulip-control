curl -sO https://www.iaik.tugraz.at/content/research/opensource/lily/lily-2008-06-19.tar.gz
echo '2442e15223e116a4ebf6fbab42646ee64111f286458b4393f18ac7b1a9e19c952bba91f548e0e9c323d50f29e639a5089bf117c5e7eae874f1e9c677f3c44aee  lily-2008-06-19.tar.gz' | \
shasum -a 512 -c -
tar xzf lily-2008-06-19.tar.gz

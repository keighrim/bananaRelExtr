JARS = \
ant.jar \
bsh.jar \
crimson.jar \
junit.jar \
jdom.jar \
salvo.jar

unpack: $(JARS)
	rm -rf $@
	mkdir $@
	for f in $(JARS) ; do \
	  (cd $@ ; jar -xf ../$$f) ; \
	done
	rm -rf $@/META-INF
	rm -rf $@/junit3.7

mallet-deps.jar: unpack
	cd $< ; \
	jar -cf ../$@ *

clean:
	rm -rf unpack
	rm -f mallet-deps.jar

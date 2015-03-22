set BASEDIR=lib\mallet_maxent
set MALLET_HOME=lib\mallet\
set MALLET_LIB=%MALLET_HOME%\lib

set CLASSPATH=%BASEDIR%\classifier\classes;%MALLET_HOME%\class;%MALLET_LIB%\mallet-deps.jar

java -classpath %CLASSPATH% -mx3000m MaxentClassifier %*


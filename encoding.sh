#!/bin/bash

if [ $# -lt 1 ]
then
  echo "Usage: "$0" <file_name>"
  echo "Convert files to utf-8"
  #exit
fi
cpt=0
for i in `find . -type f -name "*.txt"`
do
  encoding=$(file -i "$i" | cut -d ' ' -f3 | sed 's/charset=\(.*\)/\1/')
  #encoding=$(chardet $i | cut -d ' ' -f 2) #determine the encoding of current file
  #echo "encoding "$i" ("$encoding") to utf-8"
  iconv -f $encoding -t utf-8 $i -o $i.tmp #tmp file is created as a workaround for bus error
  mv $i.tmp $i
  ((cpt++))
done
echo "$cpt" " fichiers convertis en utf-8"


# code adapté de :
#https://gist.github.com/arpith20/4fcf7682a9154bc777dfcd2199edecf4

# lancer avec ./encoding.sh 2> /dev/null pour ne pas afficher les messages d'erreurs 

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19KH8xQZsyiX-4OXK5F7rFKTWwNWnRDxf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19KH8xQZsyiX-4OXK5F7rFKTWwNWnRDxf" -O images.zip && rm -rf /tmp/cookies.txt
unzip -n images.zip
mv images-master images


wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cUsL9uwqcm8i_epYZqE_QN4Xw2-aX7ru' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cUsL9uwqcm8i_epYZqE_QN4Xw2-aX7ru" -O cars.png && rm -rf /tmp/cookies.txt

mv cars.png images/
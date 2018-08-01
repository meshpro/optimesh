

default:
	@echo "make apng?"

prepare:
	for file in step*.png; do \
		convert -trim -resize 200x200 -colorspace gray -colors 4 $$file $$file; \
	done
	for file in step*.png; do \
	  optipng -quiet $$file; \
	done

apng: prepare
	# /kc is essential for the frames not to overlap
	apngasm out.png step*.png /kc /kp

gif: prepare
	# <https://stackoverflow.com/a/23285387/353337>
	convert -dispose 2 -delay 10 -loop 0 step*.png out.gif

webp: prepare
	img2webp step*.png -min_size -lossy -o out.webp

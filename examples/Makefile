

default:
	@echo "make all?"

all:
	for method in cvt-uniform-fp cpt-uniform-fp cpt-uniform-qn odt-dp-fp odt-uniform-fp odt-uniform-bfgs; do \
	  optimesh circle.vtk out.vtk -m $$method -n 50 -f "$$method-step{:03d}.png"; \
	  for file in $$method-step*.png; do \
	    convert -trim -resize 200x200 -colorspace gray -colors 4 $$file $$file; \
	    optipng -quiet $$file; \
	  done; \
	  img2webp $$method-step*.png -min_size -lossy -o $$method.webp; \
	done

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

default:
	@echo "make all?"

# if [ "$$method" == "lloyd2" ]; then \
#   optimesh circle.vtk out.vtk -m lloyd --omega 2 -n 50 -f "$$method-step{:03d}.png"; \
# elif [ "$$method" == "cvt-uniform-qnf-0.9" ]; then \
#   optimesh circle.vtk out.vtk -m cvt-uniform-qnf --omega 0.9 -n 50 -f "$$method-step{:03d}.png"; \
# else \
#   optimesh circle.vtk out.vtk -m "$$method" -n 50 -f "$$method-step{:03d}.png"; \
# fi; \

all:
	for method in \
	    cvt-uniform-qnb cvt-uniform-qnf cvt-uniform-qnf \
	    lloyd lloyd2 cpt-uniform-fp cpt-uniform-qn \
	    odt-dp-fp odt-uniform-fp odt-uniform-bfgs; do \
	  if [ "$$method" = "lloyd2" ]; then \
	    optimesh circle.xdmf out.vtk -m lloyd --omega 2 -n 50 -f "$$method-step{:03d}.png"; \
	  else \
	    optimesh circle.xdmf out.vtk -m $$method -n 50 -f "$$method-step{:03d}.png"; \
	  fi; \
	  for file in $$method-step*.png; do \
	    convert -trim -resize 200x200 $$file $$file; \
	    optipng -quiet $$file; \
	  done; \
	  img2webp $$method-step*.png -min_size -lossy -o $$method.webp; \
	done


prepare:
	for file in step*.png; do \
		convert -trim -resize 200x200 $$file $$file; \
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

clean:
	rm -f *.png *.webp

###Instructions for generating the example APNGs

* Create original Delaunay triangulation with `create_circle.py`

* Run the smoothers:
  ```
  optimesh circle.vtk out.vtk -m {method} -n 50 -f "step{:03d}.png"
  ```

* Create the animated PNG/GIF: `make apng`/`make gif`

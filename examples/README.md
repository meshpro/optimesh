### Instructions for generating the example APNGs

* Create original Delaunay triangulation with `create-circle.py`

* Run the smoothers:
  ```
  optimesh circle.vtk out.vtk -m {method} -n 50 -f "step{:03d}.png"
  ```

* Create the animated PNG/GIF: `make apng`/`make gif`


### Instructions for generating the sphere mesh optimization WebPs

* Run with the desired method
  ```python
  import meshzoo
  import optimesh

  points, cells = meshzoo.tetra_sphere(20)


  class Sphere:
      def f(self, x):
          return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

      def grad(self, x):
          return -2 * x


  points, cells = optimesh.optimize_points_cells(
      points,
      cells,
      "cpt (fixed-point)",
      1.0e-2,
      100,
      implicit_surface=Sphere(),
      step_filename_format="out{:03d}.vtk",
  )
  ```

* Open the resulting `out*` files in ParaView and _Save Animation_.

* Crop to content, resize
  ```
  for file in *png; do convert -trim -resize 200x200\! "$file" "$file"; done
  ```
  Note the `\!` that forces the `200x200` size and disrespects the aspect ratio. If not
  using the exclamation mark, sometimes the PNGs will be of size `200x199`.

* Convert to WebP
  ```
  img2webp step*.png -min_size -lossy -o out.webp
  ```

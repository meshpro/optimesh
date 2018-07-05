Instructions for generating the example gifs:

* Create original Delaunay triangulation with create_circle.py

* Run the smoothers:
  ```
  optimesh circle.vtk out.vtk -m {method} -n 50 -f "step{:03d}.png" [-u]
  ```

* Trim all PNG files:
  ```
  for file in step*.png; do convert -trim $file $file; done
  ```

* Optimize all PNG files for size:
  ```
  for file in step*.png; do optipng $file; done
  ```

* Create an APNG:
  ```
  apngasm out.apng step*.png
  ```

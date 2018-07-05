###Instructions for generating the example APNGs

* Create original Delaunay triangulation with `create_circle.py`

* Run the smoothers:
  ```
  optimesh circle.vtk out.vtk -m {method} -n 50 -f "step{:03d}.png" [-u]
  ```

* Trim and resize all PNG files:
  ```
  for file in step*.png; do convert -trim -resize 200x200 $file $file; done
  ```

* Optimize all PNG files for size:
  ```
  for file in step*.png; do optipng $file; done
  ```

* Create an APNG:
  ```
  apngasm out.apng step*.png
  ```

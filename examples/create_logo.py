import meshzoo
import meshio


points, cells = meshzoo.hexagon(1)
meshio.write_points_cells('hex.svg', points, {'triangle': cells})

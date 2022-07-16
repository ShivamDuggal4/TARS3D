import open3d as o3d

def make_pcd(point_cloud, color=None, per_vertex_color=None, estimate_normals=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    if per_vertex_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(per_vertex_color)
    
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=100))
    return pcd

def make_line_set(points, edges, line_color = None, per_line_color=None):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    if per_line_color is not None:
        line_set.colors = o3d.utility.Vector3dVector(per_line_color)
    
    return line_set

def make_mesh(vertices, faces, colors=None):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh
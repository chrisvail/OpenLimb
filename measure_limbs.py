import torch


def plane_edge_intersection(a, b, n, p):
    # print(f"\tIntersection: {a.shape=}    {b.shape=}    {n.shape=}    {p.shape=}")
    if len(a.shape) == 2:
        t = torch.einsum("ij, ij -> i", p - a, n) / torch.einsum("ij, ij -> i", b - a, n)
    elif len(a.shape) == 3:
        t = torch.einsum("kij, ij -> ki", p - a, n) / torch.einsum("kij, ij -> ki", b - a, n)
    else:
        t = torch.dot(p - a, n) / torch.dot(b - a, n)

    return torch.nan_to_num(a + t[..., None] * (b - a), nan=float('inf')), t




def measure_planar_circumference(verts, edge2vert, face2edge, plane_point, plane_normal):
    # print(f"Circumference: {verts.shape=}    {edge2vert.shape=}    {face2edge.shape=}    {plane_point.shape=}    {plane_normal.shape=}")
    if len(verts.shape) == 2:
        verts = verts[None]
    if isinstance(plane_point, int) or (isinstance(plane_point, torch.Tensor) and not len(plane_point.shape)):
        plane_point = verts[:, plane_point].unsqueeze(1)

    if isinstance(plane_normal, (tuple, list)):
        plane_normal = verts[:, plane_normal[0]] - verts[:, plane_normal[1]]
        plane_normal /= torch.linalg.norm(plane_normal)

    edge_verts = verts[:, edge2vert]
    intersections, t = plane_edge_intersection(
        edge_verts[:, :, 0], edge_verts[:, :, 1], plane_normal, plane_point
    )


    t_mask = (0 < t) * (t < 1)

    batch_size = verts.shape[0]
    
    # intersections.shape=torch.Size([8, 50334, 3])    face2edge[None].shape=torch.Size([1, 33206, 3])    faceint2.shape=torch.Size([33206, 3, 3])
    batched_face2edge = face2edge.unsqueeze(0).expand(batch_size, -1, -1)
    batch_idxs = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, face2edge.shape[0], 3)
    face_intersections = intersections[batch_idxs, batched_face2edge]

    face_distances = torch.linalg.norm(
        torch.concat(
            [
                (face_intersections[:, :, 0] - face_intersections[:, :, 1])[:, :, None],
                (face_intersections[:, :, 2] - face_intersections[:, :, 1])[:, :, None],
                (face_intersections[:, :, 0] - face_intersections[:, :, 2])[:, :, None],
            ],
            dim=2,
        ),
        dim=-1,
    )

    t_mask_faces = t_mask[batch_idxs, batched_face2edge]
    
    t_mask_faces = torch.concat(
        [
            (t_mask_faces[:, :, 0] * t_mask_faces[:, :, 1])[:, :, None],
            (t_mask_faces[:, :, 2] * t_mask_faces[:, :, 1])[:, :, None],
            (t_mask_faces[:, :, 0] * t_mask_faces[:, :, 2])[:, :, None],
        ],
        dim=2,
    )

    return torch.nansum(face_distances * t_mask_faces, dim=(1,2))



def measure_width(verts, edge2vert, plane_point, plane_normal, plane_direction):
    # print(f"Width:         {verts.shape=}    {edge2vert.shape=}    {plane_point.shape=}    {plane_normal.shape=}    {plane_direction.shape=}")

    if len(verts.shape) == 2:
        verts = verts[None]
        
    if isinstance(plane_point, int) or (isinstance(plane_point, torch.Tensor) and not len(plane_point.shape)):
        plane_point = verts[:, plane_point].unsqueeze(1)

    if isinstance(plane_normal, (tuple, list)):
        plane_normal = verts[:, plane_normal[0]] - verts[:, plane_normal[1]].unsqueeze(1)
        plane_normal /= torch.linalg.norm(plane_normal, dim=-1)

    if isinstance(plane_direction, (tuple, list)):
        plane_direction = verts[:, plane_direction[0]] - verts[:, plane_direction[1]]
        plane_direction /= torch.linalg.norm(plane_direction, dim=-1).unsqueeze(1)

    batch_size = verts.shape[0]
    batched_edge2vert = edge2vert.unsqueeze(0).expand(batch_size, -1, -1)
    batch_idxs = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, edge2vert.shape[0], 2)


    edge_verts = verts[batch_idxs, batched_edge2vert]
    intersections, t = plane_edge_intersection(
        edge_verts[:, :, 0], edge_verts[:, :, 1], plane_normal, plane_point
    )

    t_mask = (0 < t) * (t < 1)
    values = torch.einsum("kij, ij -> ki", intersections, plane_direction)
    values -= torch.min(torch.nan_to_num(values*t_mask, 1E24, 1E24, 1E24), dim=1, keepdim=True)[0]
    values = torch.nan_to_num(values, 0, 0, 0)
    values *= t_mask

    return torch.max(values, dim=1)[0] - torch.min(values, dim=1)[0]



def measure_length(verts, v1, v2, direction):
    # print(f"Length:        {verts.shape=}    {v1}    {v2}    {direction.shape=}")

    if len(verts.shape) == 2:
        verts = verts[None]
        
    if isinstance(direction, (tuple, list)):
        direction = verts[direction[0]] - verts[direction[1]]
        direction /= torch.linalg.norm(direction)

    return torch.abs(torch.einsum("ij, ij -> i", verts[:, v1] - verts[:, v2], direction))



def remove_bones(verts, path="/"):
    mapping = torch.load(path + "vert_mapping.pt")
    face2vert = torch.load(path + "face2vert.pt")

    return verts[mapping], face2vert


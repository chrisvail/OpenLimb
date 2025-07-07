import torch


def plane_edge_intersection(a, b, n, p):
    if len(a.shape) == 2:
        t = torch.einsum("ij, ij -> i", p - a, n) / torch.einsum("ij, ij -> i", b - a, n)
        t = t.unsqueeze(-1)
    else:
        t = torch.dot(p - a, n) / torch.dot(b - a, n)
    return torch.nan_to_num(a + t * (b - a), nan=float('inf')), t




def measure_plane(verts, edge2vert, face2edge, plane_point, plane_normal):
    edge_verts = verts[edge2vert]
    intersections, t = plane_edge_intersection(
        edge_verts[:, 0], edge_verts[:, 1], plane_normal, plane_point
    )

    t_mask = torch.logical_and(0 <= t, t < 1)
    print(f"{torch.sum(torch.isnan(intersections))=}    {torch.sum(t_mask)=}    {t_mask.shape=}")

    face_intersections = intersections[face2edge]
    face_distances = torch.linalg.norm(
        torch.concat(
            [
                (face_intersections[:, 0] - face_intersections[:, 1])[:, None],
                (face_intersections[:, 2] - face_intersections[:, 1])[:, None],
                (face_intersections[:, 0] - face_intersections[:, 2])[:, None],
            ],
            dim=1,
        ),
        dim=-1,
        keepdim=True,
    )

    t_mask_faces = t_mask[face2edge]
    t_mask_faces = torch.concat(
        [
            (t_mask_faces[:, 0] + t_mask_faces[:, 1] == 2)[:, None],
            (t_mask_faces[:, 2] + t_mask_faces[:, 1] == 2)[:, None],
            (t_mask_faces[:, 0] + t_mask_faces[:, 2] == 2)[:, None],
        ],
        dim=1,
    )

    print(f"{t_mask_faces.shape=}    {torch.sum(t_mask_faces)}")

    return torch.sum(face_distances * t_mask_faces)




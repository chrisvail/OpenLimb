import torch


def measure_plane(verts, edge2vert, face2edge, plane_point, plane_normal):
    edge_verts = verts[edge2vert]
    intersections, t = plane_edge_intersection(
        edge_verts[:, 0], edge_verts[:, 1], plane_normal, plane_point
    )
    t_mask = torch.logical_and(0 <= t, t < 1)

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

    return torch.sum(face_distances * t_mask_faces)


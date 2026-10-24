"""A query-point Voronoi cell against periodic framework atoms only.

No mobile ions or periodic images of the artificial query are added. A cell is
accepted only when omitted atoms cannot cut it: search radius > 2*vertex radius.
"""
import numpy as np
from scipy.spatial import Voronoi, QhullError
from pymatgen.analysis.local_env import solid_angle, vol_tetra

from .occupied_sites import site_has_element
from .voronoi_coordination_config import VoronoiCoordinationConfig


class FrameworkVoronoiError(ValueError):
    """A complete finite framework cell could not be established."""


def _ordered_face(vertices, normal):
    center = vertices.mean(axis=0)
    offsets = vertices-center
    first = offsets[np.argmax(np.linalg.norm(offsets,axis=1))]
    first = first/np.linalg.norm(first)
    second = np.cross(normal,first)
    order = np.argsort(np.arctan2(offsets@second,offsets@first))
    return vertices[order]


def framework_voronoi_neighbors(structure, frac_point, probe_element='Li', config=None):
    settings = config or VoronoiCoordinationConfig()
    if not len(structure) or not structure.is_ordered:
        raise ValueError('A nonempty ordered structure is required')
    point = np.asarray(frac_point,dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Supply three finite fractional coordinates')
    center = structure.lattice.get_cartesian_coords(point % 1.)
    cutoff = settings.initial_cutoff_A
    last_reason = 'No closed cell'
    while True:
        sites = [site for site in structure.get_sites_in_sphere(center,cutoff)
                 if not site_has_element(site,probe_element)]
        vectors = np.asarray([site.coords-center for site in sites],dtype=float).reshape(-1,3)
        distances = np.linalg.norm(vectors,axis=1)
        if np.any(distances <= settings.numeric_tolerance):
            raise FrameworkVoronoiError('Query coincides with a framework atom')
        try:
            if len(sites) < 4:
                raise FrameworkVoronoiError('Too few framework images')
            voronoi = Voronoi(np.vstack([np.zeros(3),vectors]))
            region = voronoi.regions[voronoi.point_region[0]]
            if not region or -1 in region:
                raise FrameworkVoronoiError('Unbounded query cell')
            vertex_radius = float(np.linalg.norm(voronoi.vertices[region],axis=1).max())
            if 2*vertex_radius + settings.numeric_tolerance >= cutoff:
                raise FrameworkVoronoiError('Search radius does not certify complete cell')
            neighbors = []
            for pair,indices in voronoi.ridge_dict.items():
                if 0 not in pair:
                    continue
                if -1 in indices:
                    raise FrameworkVoronoiError('Infinite face')
                other = pair[0] if pair[1] == 0 else pair[1]
                local = other-1
                normal = vectors[local]/distances[local]
                face = _ordered_face(voronoi.vertices[indices],normal)
                volume = sum(vol_tetra(np.zeros(3),face[0],face[j],face[j+1])
                             for j in range(1,len(face)-1))
                area = float(3*volume/(distances[local]/2))
                if area <= settings.minimum_face_area_A2:
                    continue
                site = sites[local]
                neighbors.append({
                    'site_index':int(site.index),'element':site.specie.symbol,
                    'distance':float(distances[local]),'solid_angle':float(solid_angle(np.zeros(3),face)),
                    'area':area,'volume':float(volume),'normal':normal,
                    'image':tuple(int(value) for value in site.image),
                })
            total_angle = sum(n['solid_angle'] for n in neighbors)
            if not neighbors or not np.isclose(total_angle,4*np.pi,rtol=settings.solid_angle_sum_tolerance,
                                               atol=settings.solid_angle_sum_tolerance):
                raise FrameworkVoronoiError('Cell solid-angle sum is not 4*pi')
            return neighbors, {'cutoff_A':cutoff,'maximum_vertex_radius_A':vertex_radius,
                               'solid_angle_sum':total_angle,'complete_cell':True}
        except (QhullError,FrameworkVoronoiError) as error:
            last_reason = str(error)
        if cutoff >= settings.maximum_cutoff_A:
            break
        cutoff = min(cutoff*settings.cutoff_growth,settings.maximum_cutoff_A)
    raise FrameworkVoronoiError(f'No certified cell by {cutoff:g} A: {last_reason}')

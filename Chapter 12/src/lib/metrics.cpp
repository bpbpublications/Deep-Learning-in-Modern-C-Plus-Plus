#include "book/metrics.hpp"

TYPE intersection_over_union(const BoundingBox &A, const BoundingBox &B) {
    TYPE result = TYPE(0.);

    // calculating the intersection area
    TYPE xA = std::max(A.xmin, B.xmin);
    TYPE yA = std::max(A.ymin, B.ymin);
    TYPE xB = std::min(A.xmax, B.xmax);
    TYPE yB = std::min(A.ymax, B.ymax);
    TYPE intersection_area = std::max(TYPE(0.), xB - xA) * std::max(TYPE(0.), yB - yA);

    // Calculating union
    TYPE area_A = (A.xmax - A.xmin) * (A.ymax - A.ymin);
    TYPE area_B = (B.xmax - B.xmin) * (B.ymax - B.ymin);
    TYPE union_area = area_A + area_B - intersection_area;
    
    if (union_area > TYPE(0.)) {
        result = intersection_area / union_area;
    }

    return result;
}
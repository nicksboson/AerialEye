# Single-Image 3D Building Height Estimation (Shadow-Based)

This module extends the existing detection pipeline by reusing already-detected building boxes/masks and estimating approximate building heights from shadows visible in the same analyzed image.

## Pipeline Summary
1. Reuse existing building detections (`detect_buildings()`).
2. Extract shadow candidates from grayscale + Gaussian blur + adaptive threshold (`detect_shadows()`).
3. Associate the nearest plausible shadow contour to each building.
4. Compute longest shadow direction from building edge (`compute_shadow_length()`).
5. Estimate building height using:

`H = S * tan(theta)`

Where:
- `H` = estimated building height (meters)
- `S` = shadow length (meters)
- `theta` = solar elevation angle (degrees)

6. Draw overlays (`draw_height_overlay()`) and store:
- `analyzed_output.jpg`
- `results.json`

## Default Parameters
- `meters_per_pixel = 0.2`
- `solar_elevation_angle = 45`

## Assumptions
- Shadows are visible and not completely occluded.
- Buildings already detected by existing pipeline are valid.
- Meter-per-pixel is reasonably calibrated for the source imagery.
- Solar elevation angle is available or approximated.

## Limitations and Constraints
- Single-image monocular shadow analysis is an approximation.
- Accuracy is sensitive to:
  - sun direction and solar elevation angle errors
  - low contrast / cloud cover
  - shadow overlap from nearby structures/trees
  - orthorectification and perspective distortions
- Top-view satellite scenes with weak cast shadows may under-estimate heights.

## Future Improvements
- Stereo imagery / multi-view photogrammetry for direct 3D triangulation.
- WebODM integration for orthophoto + dense cloud + DSM/DTM workflows.
- DSM/CHM-based absolute height estimation.
- Learned depth-estimation models fused with shadow geometry.
- Temporal stabilization using tracked multi-frame building IDs (video path).

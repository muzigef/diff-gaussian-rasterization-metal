/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                          -1.0925484305920792f, 0.5462742152960396f};
constant float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                          -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

// MSL adaptations of upstream analytic derivatives. Matrix arguments remain column-major.
float3x3 mat3(float a, float b, float c, float d, float e, float f, float g, float h, float i) {
    return float3x3(float3(a, b, c), float3(d, e, f), float3(g, h, i));
}
float3x3 mat3(float diagonal) {
    return float3x3(diagonal);
}
float3 transformPoint4x3(float3 p, device const float *m) {
    return float3(transform(m, p, 0), transform(m, p, 1), transform(m, p, 2));
}
float4 transformPoint4x4(float3 p, device const float *m) {
    return float4(transform(m, p, 0), transform(m, p, 1), transform(m, p, 2), transform(m, p, 3));
}
float3 transformVec4x3Transpose(float3 p, device const float *m) {
    return float3(m[0] * p.x + m[1] * p.y + m[2] * p.z, m[4] * p.x + m[5] * p.y + m[6] * p.z,
                  m[8] * p.x + m[9] * p.y + m[10] * p.z);
}

float3 dnormvdv(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

    float3 dnormvdv;
    dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdv;
}

float3 forward_sh(int idx, int deg, int max_coeffs, device const packed_float3 *means, float3 campos,
                  device const float *shs, device bool *clamped) {
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    float3 pos = means[idx];
    float3 dir = pos - campos;
    dir = dir / length(dir);

    device const packed_float3 *sh = ((device const packed_float3 *)shs) + idx * max_coeffs;
    float3 result = SH_C0 * sh[0];

    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return max(result, 0.0f);
}

void forward_cov(float3 scale, float mod, float4 rot, device float *cov3D) {
    // Create scaling matrix
    float3x3 S = mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    float4 q = rot; // / length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    float3x3 R = mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                      2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    float3x3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    float3x3 Sigma = transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

void backward_sh(int idx, int deg, int max_coeffs, device const packed_float3 *means, float3 campos,
                 device const float *shs, device const bool *clamped, device const packed_float3 *dL_dcolor,
                 device packed_float3 *dL_dmeans, device packed_float3 *dL_dshs) {
    // Compute intermediate values, as it is done during forward
    float3 pos = means[idx];
    float3 dir_orig = pos - campos;
    float3 dir = dir_orig / length(dir_orig);

    device const packed_float3 *sh = ((device const packed_float3 *)shs) + idx * max_coeffs;

    // Use PyTorch rule for clamping: if clamping was applied,
    // gradient becomes 0.
    float3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    float3 dRGBdx(0, 0, 0);
    float3 dRGBdy(0, 0, 0);
    float3 dRGBdz(0, 0, 0);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Target location for this Gaussian to write SH gradients to
    device packed_float3 *dL_dsh = dL_dshs + idx * max_coeffs;

    // No tricks here, just high school-level calculus.
    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    if (deg > 0) {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] +
                      SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] +
                      SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2) {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy + SH_C3[1] * sh[10] * yz +
                           SH_C3[2] * sh[11] * -2.f * xy + SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                           SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) + SH_C3[5] * sh[14] * 2.f * xz +
                           SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) + SH_C3[1] * sh[10] * xz +
                           SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                           SH_C3[3] * sh[12] * -3.f * 2.f * yz + SH_C3[4] * sh[13] * -2.f * xy +
                           SH_C3[5] * sh[14] * -2.f * yz + SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (SH_C3[1] * sh[10] * xy + SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                           SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                           SH_C3[4] * sh[13] * 4.f * 2.f * xz + SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    // The view direction is an input to the computation. View direction
    // is influenced by the Gaussian's mean, so SHs gradients
    // must propagate back into 3D position.
    float3 dL_ddir(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));

    // Account for normalization of direction
    float3 dL_dmean =
        dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

    // Gradients of loss w.r.t. Gaussian means, but only the portion
    // that is caused because the mean affects the view-dependent color.
    // Additional mean gradient is accumulated in below methods.
    dL_dmeans[idx] += float3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

void backward_cov2d(uint idx, device const packed_float3 *means, device const int *radii,
                    device const float *cov3Ds, float h_x, float h_y, float tan_fovx, float tan_fovy,
                    device const float *view_matrix, device const float *dL_dconics,
                    device packed_float3 *dL_dmeans, device float *dL_dcov) {

    if (!(radii[idx] > 0))
        return;

    // Reading location of 3D covariance for this Gaussian
    device const float *cov3D = cov3Ds + 6 * idx;

    // Fetch gradients, recompute 2D covariance and relevant
    // intermediate forward results needed in the backward.
    float3 mean = means[idx];
    float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]};
    float3 t = transformPoint4x3(mean, view_matrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    float3x3 J = mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z), 0.0f, h_y / t.z,
                      -(h_y * t.y) / (t.z * t.z), 0, 0, 0);

    float3x3 W = mat3(view_matrix[0], view_matrix[4], view_matrix[8], view_matrix[1], view_matrix[5],
                      view_matrix[9], view_matrix[2], view_matrix[6], view_matrix[10]);

    float3x3 Vrk =
        mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

    float3x3 T = W * J;

    float3x3 cov2D = transpose(T) * transpose(Vrk) * T;

    // Use helper variables for 2D covariance entries. More compact.
    float a = cov2D[0][0] += 0.3f;
    float b = cov2D[0][1];
    float c = cov2D[1][1] += 0.3f;

    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;
    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    if (denom2inv != 0) {
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db =
            denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (diagonal).
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov[6 * idx + 0] =
            (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
        dL_dcov[6 * idx + 3] =
            (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
        dL_dcov[6 * idx + 5] =
            (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (off-diagonal).
        // Off-diagonal elements appear twice --> double the gradient.
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da +
                               (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db +
                               2 * T[1][0] * T[1][1] * dL_dc;
        dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da +
                               (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db +
                               2 * T[1][0] * T[1][2] * dL_dc;
        dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da +
                               (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db +
                               2 * T[1][1] * T[1][2] * dL_dc;
    } else {
        for (int i = 0; i < 6; i++)
            dL_dcov[6 * idx + i] = 0;
    }

    // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
    // cov2D = transpose(T) * transpose(Vrk) * T;
    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
                    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
    float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
                    (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
    float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
                    (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
    float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
                    (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
    float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
                    (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
    float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
                    (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

    // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
    // T = W * J
    float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    float tz = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    // Gradients of loss w.r.t. transformed Gaussian mean t
    float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
    float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
    float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 +
                   (2 * h_y * t.y) * tz3 * dL_dJ12;

    // Account for transformation of mean to t
    // t = transformPoint4x3(mean, view_matrix);
    float3 dL_dmean = transformVec4x3Transpose({dL_dtx, dL_dty, dL_dtz}, view_matrix);

    // Gradients of loss w.r.t. Gaussian means, but only the portion
    // that is caused because the mean affects the covariance matrix.
    // Additional mean gradient is accumulated in BACKWARD::preprocess.
    dL_dmeans[idx] = dL_dmean;
}

void backward_cov3d(int idx, float3 scale, float mod, float4 rot, device const float *dL_dcov3Ds,
                    device packed_float3 *dL_dscales, device float4 *dL_drots) {
    // Recompute (intermediate) results for the 3D covariance computation.
    float4 q = rot; // / length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    float3x3 R = mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                      2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    float3x3 S = mat3(1.0f);

    float3 s = mod * scale;
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    float3x3 M = S * R;

    device const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

    float3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
    float3 ounc = 0.5f * float3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

    // Convert per-element covariance loss gradients to matrix form
    float3x3 dL_dSigma =
        mat3(dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[1], dL_dcov3D[3],
             0.5f * dL_dcov3D[4], 0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

    // Compute loss gradient w.r.t. matrix M
    // dSigma_dM = 2 * M
    float3x3 dL_dM = 2.0f * M * dL_dSigma;

    float3x3 Rt = transpose(R);
    float3x3 dL_dMt = transpose(dL_dM);

    // Gradients of loss w.r.t. scale
    device packed_float3 *dL_dscale = dL_dscales + idx;
    dL_dscale->x = dot(Rt[0], dL_dMt[0]);
    dL_dscale->y = dot(Rt[1], dL_dMt[1]);
    dL_dscale->z = dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    // Gradients of loss w.r.t. normalized quaternion
    float4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
              2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
              2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
              2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
              2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    // Gradients of loss w.r.t. unnormalized quaternion
    device float4 *dL_drot = (device float4 *)(dL_drots + idx);
    *dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w },
                                                           // float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

struct TrainingParams {
    uint4 dimensions;
    uint4 modes;
    float4 optics;
};

kernel void training_camera(device const float *view [[buffer(0)]], device const float *proj [[buffer(1)]],
                            device const float *bg [[buffer(2)]], device const float *campos [[buffer(3)]],
                            device float *camera [[buffer(4)]], constant TrainingParams &p [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id != 0)
        return;
    for (uint i = 0; i < 16; i++) {
        camera[i] = view[i];
        camera[16 + i] = proj[i];
    }
    camera[32] = p.dimensions.y;
    camera[33] = p.dimensions.z;
    camera[34] = p.dimensions.y / (2 * p.optics.x);
    camera[35] = p.dimensions.z / (2 * p.optics.y);
    camera[36] = p.optics.x;
    camera[37] = p.optics.y;
    for (uint i = 0; i < 3; i++) {
        camera[38 + i] = bg[i];
        camera[41 + i] = campos[i];
    }
    camera[44] = p.optics.z;
}

kernel void
training_prepare(device const packed_float3 *means [[buffer(0)]], device const float *colors [[buffer(1)]],
                 device const float *opacity [[buffer(2)]], device const packed_float3 *scales [[buffer(3)]],
                 device const float4 *rotations [[buffer(4)]], device const float *cov [[buffer(5)]],
                 device const float *sh [[buffer(6)]], device const float *camera [[buffer(7)]],
                 device float *packed [[buffer(8)]], device bool *clamped [[buffer(9)]],
                 device float *saved_cov [[buffer(10)]], device atomic_uint *error [[buffer(11)]],
                 constant TrainingParams &p [[buffer(12)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.dimensions.x)
        return;
    float3 mean = means[id];
    for (uint i = 0; i < 3; i++)
        packed[id * 13 + i] = mean[i];
    packed[id * 13 + 12] = opacity[id];
    if (transform(camera, mean, 2) <= 0.2f) {
        if (p.modes.w)
            atomic_store_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    if (p.modes.z)
        forward_cov(scales[id], p.optics.z, rotations[id], saved_cov + id * 6);
    else
        for (uint i = 0; i < 6; i++)
            saved_cov[id * 6 + i] = cov[id * 6 + i];
    for (uint i = 0; i < 6; i++)
        packed[id * 13 + 3 + i] = saved_cov[id * 6 + i];
    float3 rgb;
    if (p.modes.y)
        rgb = forward_sh(id, p.modes.x, p.dimensions.w, means, float3(camera[41], camera[42], camera[43]), sh,
                         clamped);
    else
        rgb = float3(colors[id * 3], colors[id * 3 + 1], colors[id * 3 + 2]);
    for (uint i = 0; i < 3; i++)
        packed[id * 13 + 9 + i] = rgb[i];
}

kernel void training_radii(device const Projected *projected [[buffer(0)]], device int *radii [[buffer(1)]],
                           constant uint4 &p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < p.x)
        radii[id] = int(projected[id].centerDepthRadius.w);
}

kernel void training_render(device const float *packed [[buffer(0)]],
                            device const Projected *projected [[buffer(1)]],
                            device const uint4 *records [[buffer(2)]],
                            device const uint2 *ranges [[buffer(3)]], device float *finalT [[buffer(4)]],
                            device uint *lastContributor [[buffer(5)]],
                            device const float *camera [[buffer(6)]], device float *output [[buffer(7)]],
                            constant uint4 &p [[buffer(8)]], uint pixel [[thread_position_in_grid]]) {
    uint size = p.x * p.y;
    if (pixel >= size)
        return;
    uint2 xy(pixel % p.x, pixel / p.x);
    uint2 range = ranges[(xy.y / 16) * p.z + xy.x / 16];
    float t = 1;
    float3 rgb(0);
    uint last = 0;
    for (uint index = range.x; index < range.y; index++) {
        uint id = records[index].z;
        Projected g = projected[id];
        float2 d = g.centerDepthRadius.xy - float2(xy);
        float4 co = g.conicOpacity;
        float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
        if (power > 0)
            continue;
        float alpha = min(0.99f, co.w * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;
        float next = t * (1 - alpha);
        if (next < 0.0001f)
            break;
        rgb += float3(packed[id * 13 + 9], packed[id * 13 + 10], packed[id * 13 + 11]) * alpha * t;
        t = next;
        last = index - range.x + 1;
    }
    finalT[pixel] = t;
    lastContributor[pixel] = last;
    rgb += t * float3(camera[38], camera[39], camera[40]);
    for (uint ch = 0; ch < 3; ch++)
        output[ch * size + pixel] = rgb[ch];
}

void add_float(device float *address, float value) {
    device atomic_uint *ptr = (device atomic_uint *)address;
    uint old = atomic_load_explicit(ptr, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(ptr, &old, as_type<uint>(as_type<float>(old) + value),
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
}

kernel void training_backward_render(
    device const float *packed [[buffer(0)]], device const Projected *projected [[buffer(1)]],
    device const uint4 *records [[buffer(2)]], device const uint2 *ranges [[buffer(3)]],
    device const float *finalT [[buffer(4)]], device const uint *lastContributor [[buffer(5)]],
    device const float *camera [[buffer(6)]], device const float *grad_pixels [[buffer(7)]],
    device float *grad_mean2d [[buffer(8)]], device float *grad_conic [[buffer(9)]],
    device float *grad_opacity [[buffer(10)]], device float *grad_colors [[buffer(11)]],
    constant uint4 &p [[buffer(12)]], uint pixel [[thread_position_in_grid]]) {
    uint size = p.x * p.y;
    if (pixel >= size)
        return;
    uint2 xy(pixel % p.x, pixel / p.x);
    uint2 range = ranges[(xy.y / 16) * p.z + xy.x / 16];
    float t_final = finalT[pixel], t = t_final;
    float3 dp(grad_pixels[pixel], grad_pixels[size + pixel], grad_pixels[2 * size + pixel]);
    float3 accum_rec(0), last_color(0);
    float last_alpha = 0;
    uint contributor = range.y - range.x;
    for (uint position = range.y; position > range.x;) {
        uint id = records[--position].z;
        if (--contributor >= lastContributor[pixel])
            continue;
        Projected g = projected[id];
        float4 co = g.conicOpacity;
        float2 d = g.centerDepthRadius.xy - float2(xy);
        float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
        if (power > 0)
            continue;
        float G = exp(power), alpha = min(0.99f, co.w * G);
        if (alpha < 1.0f / 255.0f)
            continue;
        t = t / (1 - alpha);
        float d_alpha = 0;
        for (uint ch = 0; ch < 3; ch++) {
            float c = packed[id * 13 + 9 + ch];
            accum_rec[ch] = last_alpha * last_color[ch] + (1 - last_alpha) * accum_rec[ch];
            last_color[ch] = c;
            d_alpha += (c - accum_rec[ch]) * dp[ch];
            add_float(grad_colors + id * 3 + ch, alpha * t * dp[ch]);
        }
        d_alpha *= t;
        last_alpha = alpha;
        float bg_dot = 0;
        for (uint ch = 0; ch < 3; ch++)
            bg_dot += camera[38 + ch] * dp[ch];
        d_alpha += (-t_final / (1 - alpha)) * bg_dot;
        float dG = co.w * d_alpha, gdx = G * d.x, gdy = G * d.y;
        add_float(grad_mean2d + id * 3, dG * (-gdx * co.x - gdy * co.y) * (0.5f * p.x));
        add_float(grad_mean2d + id * 3 + 1, dG * (-gdy * co.z - gdx * co.y) * (0.5f * p.y));
        add_float(grad_conic + id * 4, -0.5f * gdx * d.x * dG);
        add_float(grad_conic + id * 4 + 1, -0.5f * gdx * d.y * dG);
        add_float(grad_conic + id * 4 + 3, -0.5f * gdy * d.y * dG);
        add_float(grad_opacity + id, G * d_alpha);
    }
}

kernel void training_backward_preprocess(
    device const packed_float3 *means [[buffer(0)]], device const int *radii [[buffer(1)]],
    device const float *sh [[buffer(2)]], device const bool *clamped [[buffer(3)]],
    device const packed_float3 *scales [[buffer(4)]], device const float4 *rotations [[buffer(5)]],
    device const float *saved_cov [[buffer(6)]], device const float *camera [[buffer(7)]],
    device const packed_float3 *grad_mean2d [[buffer(8)]], device const float *grad_conic [[buffer(9)]],
    device packed_float3 *grad_mean3d [[buffer(10)]], device float *grad_colors [[buffer(11)]],
    device float *grad_cov [[buffer(12)]], device float *grad_sh [[buffer(13)]],
    device packed_float3 *grad_scales [[buffer(14)]], device float4 *grad_rotations [[buffer(15)]],
    constant TrainingParams &p [[buffer(16)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.dimensions.x || radii[id] <= 0)
        return;
    backward_cov2d(id, means, radii, saved_cov, camera[34], camera[35], p.optics.x, p.optics.y, camera,
                   grad_conic, grad_mean3d, grad_cov);
    float3 m = means[id];
    device const float *proj = camera + 16;
    float4 hom = transformPoint4x4(m, proj);
    float invw = 1 / (hom.w + 0.0000001f);
    float mul1 = hom.x * invw * invw, mul2 = hom.y * invw * invw;
    float3 dm;
    for (uint k = 0; k < 3; k++)
        dm[k] = (proj[k * 4] * invw - proj[k * 4 + 3] * mul1) * grad_mean2d[id].x +
                (proj[k * 4 + 1] * invw - proj[k * 4 + 3] * mul2) * grad_mean2d[id].y;
    grad_mean3d[id] += dm;
    if (p.modes.y)
        backward_sh(id, p.modes.x, p.dimensions.w, means, float3(camera[41], camera[42], camera[43]), sh,
                    clamped, (device const packed_float3 *)grad_colors, grad_mean3d,
                    (device packed_float3 *)grad_sh);
    if (p.modes.z)
        backward_cov3d(id, scales[id], p.optics.z, rotations[id], grad_cov, grad_scales, grad_rotations);
}

kernel void training_visible(device const packed_float3 *means [[buffer(0)]],
                             device const float *view [[buffer(1)]], device bool *visible [[buffer(2)]],
                             constant uint4 &p [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id < p.x)
        visible[id] = transform(view, means[id], 2) > 0.2f;
}

using UnityEngine;

public static class BallisticUtility
{
    public static bool EvaluateShot(
        Vector3 targetFromMuzzleInRobotFrame,
        float yawDeg,
        float pitchDeg,
        float rpm,
        Vector2 chassisVelocityXZ,
        out float missDistance
    )
    {
        const float gravity = 9.81f;
        const float wheelRadiusMeters = 0.0508f;
        const float hitRadiusMeters = 0.2f;

        float exitSpeed = rpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        float yawRad = yawDeg * Mathf.Deg2Rad;
        float pitchRad = pitchDeg * Mathf.Deg2Rad;

        float horizontalSpeed = exitSpeed * Mathf.Cos(pitchRad);

        float projectileVx = horizontalSpeed * Mathf.Sin(yawRad) + chassisVelocityXZ.x;
        float projectileVz = horizontalSpeed * Mathf.Cos(yawRad) + chassisVelocityXZ.y;
        float projectileVy = exitSpeed * Mathf.Sin(pitchRad);

        float targetX = targetFromMuzzleInRobotFrame.x;
        float targetY = targetFromMuzzleInRobotFrame.y;
        float targetZ = targetFromMuzzleInRobotFrame.z;

        float a = 0.5f * gravity;
        float b = -projectileVy;
        float c = targetY;

        float discriminant = b * b - 4f * a * c;

        missDistance = 10f;

        if (discriminant < 0f)
        {
            float maxHeight = (projectileVy * projectileVy) / (2f * gravity);
            missDistance = 5f + Mathf.Abs(targetY - maxHeight);
            return false;
        }

        float sqrtD = Mathf.Sqrt(discriminant);

        float t1 = (-b - sqrtD) / (2f * a);
        float t2 = (-b + sqrtD) / (2f * a);

        bool foundValidTime = false;
        float bestMiss = float.MaxValue;

        if (t1 > 0f)
        {
            float x = projectileVx * t1;
            float z = projectileVz * t1;
            float miss = Mathf.Sqrt(Mathf.Pow(x - targetX, 2f) + Mathf.Pow(z - targetZ, 2f));
            bestMiss = Mathf.Min(bestMiss, miss);
            foundValidTime = true;
        }

        if (t2 > 0f)
        {
            float x = projectileVx * t2;
            float z = projectileVz * t2;
            float miss = Mathf.Sqrt(Mathf.Pow(x - targetX, 2f) + Mathf.Pow(z - targetZ, 2f));
            bestMiss = Mathf.Min(bestMiss, miss);
            foundValidTime = true;
        }

        if (!foundValidTime)
        {
            missDistance = 10f;
            return false;
        }

        missDistance = bestMiss;
        return missDistance <= hitRadiusMeters;
    }
}
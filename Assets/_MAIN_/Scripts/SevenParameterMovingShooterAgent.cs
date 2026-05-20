using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class SevenParameterMovingShooterAgent : Agent
{
    public enum TargetCorner
    {
        PositiveXPositiveZ,
        NegativeXPositiveZ,
        PositiveXNegativeZ,
        NegativeXNegativeZ
    }

    [Header("Scene References")]
    public Transform fieldCenter;
    public Transform robotRoot;
    public Transform turretYawPivot;
    public Transform hoodPitchPivot;
    public Transform muzzle;
    public Transform target;
    public Rigidbody ballPrefab;

    [Header("FTC Field")]
    public float fieldSizeMeters = 3.6576f; // 12 ft
    public TargetCorner targetCorner = TargetCorner.PositiveXPositiveZ;
    public bool autoPlaceTargetAtCorner = true;

    [Tooltip("Keep this FALSE if you want yaw to matter. If true, robot body faces target corner and yaw becomes easier.")]
    public bool autoFaceRobotTowardTargetCorner = false;

    [Header("Document Target Ranges")]
    public float minTargetForward = 0.5f;
    public float maxTargetForward = 3.5f;
    public float maxTargetSide = 1.5f;
    public float targetRelativeHeight = 1.2f;

    [Header("Robot Motion A -> B")]
    public float fixedMoveSpeed = 1.5f;
    public float minPathDistance = 0.75f;

    [Tooltip("FTC-style mecanum behavior: robot translates but keeps heading fixed.")]
    public bool keepRobotHeadingFixed = true;

    [Header("A/B Point Validation")]
    public bool validateReachability = true;
    public int maxPointSearchAttempts = 5000;
    public float minActualForwardAtShoot = 0.5f;
    public float maxActualForwardAtShoot = 3.5f;
    public float maxActualSideAtShoot = 1.5f;

    [Header("Yaw Training Constraints")]
    public float minAbsSideOffsetAtShoot = 0.25f;
    public float minLateralVelocityForYawTraining = 0.25f;

    [Header("Turret Axis Settings")]
    public Vector3 yawLocalAxis = Vector3.up;
    public Vector3 pitchLocalAxis = Vector3.up;
    public float yawDirection = 1f;
    public float pitchDirection = -1f;

    [Header("Mechanical Limits From Document")]
    public float minYawDeg = -90f;
    public float maxYawDeg = 90f;
    public float minPitchDeg = 0f;
    public float maxPitchDeg = 90f;
    public float maxYawDeltaPerStep = 5f;
    public float maxPitchDeltaPerStep = 2f;
    public float maxRpm = 6000f;

    [Header("Shooting / Reward")]
    public int aimStepsBeforeShot = 20;
    public float hitRadius = 0.2f;
    public float hitReward = 100f;
    public float timePenalty = -0.1f;
    public float missPenaltyMultiplier = 1f;

    [Header("Visual Mode")]
    public bool spawnVisualBall = false;
    public bool delayEpisodeForVisuals = false;
    public float visualSecondsAfterShot = 1.5f;
    public bool drawDebugTrajectory = false;
    public float debugTrajectorySeconds = 2.0f;
    public int debugTrajectorySegments = 30;
    public float debugTrajectoryDuration = 0.25f;

    [Header("Observation Normalization")]
    public float positionNormalizer = 3.5f;
    public float velocityNormalizer = 2f;

    [Header("Evaluation / Accuracy Test")]
    public bool evaluationMode = false;
    public int evaluationEpisodeLimit = 1000;
    public bool stopAfterEvaluation = true;
    public bool logEveryEpisode = true;

    private int evalEpisodesCompleted = 0;
    private int evalHits = 0;
    private int evalMisses = 0;
    private float evalTotalMissDistance = 0f;
    private float evalStartTime = 0f;
    private bool evaluationStarted = false;

    private Vector3 pointA;
    private Vector3 pointB;
    private Vector3 moveDirectionWorld;
    private Vector3 chassisVelocityWorld;

    private float currentYawDeg;
    private float currentPitchDeg;
    private float currentRpm;

    private Quaternion robotBaseRotation;
    private Quaternion yawBaseRotation;
    private Quaternion pitchBaseRotation;

    private bool hasShot;
    private bool waitingForVisualEnd;
    private float visualTimer;
    private Rigidbody activeBall;

    private bool warnedTargetHeight = false;

    private void Awake()
    {
        yawBaseRotation = turretYawPivot.localRotation;
        pitchBaseRotation = hoodPitchPivot.localRotation;

        ConfigureFieldTargetAndRobotHeading();
    }

    public override void OnEpisodeBegin()
    {
        ConfigureFieldTargetAndRobotHeading();

        if (evaluationMode && !evaluationStarted)
        {
            evaluationStarted = true;
            evalStartTime = Time.realtimeSinceStartup;
            Time.timeScale = 1f;
        }

        CleanupBall();

        hasShot = false;
        waitingForVisualEnd = false;
        visualTimer = 0f;

        currentYawDeg = 0f;
        currentPitchDeg = 45f;
        currentRpm = maxRpm * 0.7f;

        if (keepRobotHeadingFixed)
        {
            robotRoot.rotation = robotBaseRotation;
        }

        ApplyTurretPose();

        GenerateValidPointAAndB();

        robotRoot.position = pointA;

        Vector3 flatDirection = pointB - pointA;
        flatDirection.y = 0f;

        moveDirectionWorld = flatDirection.normalized;
        chassisVelocityWorld = moveDirectionWorld * fixedMoveSpeed;

        if (keepRobotHeadingFixed)
        {
            robotRoot.rotation = robotBaseRotation;
        }

        ApplyTurretPose();
        ValidateTargetHeightOnce();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 targetFromMuzzleWorld = target.position - muzzle.position;
        Vector3 targetFromMuzzleRobot = robotRoot.InverseTransformDirection(targetFromMuzzleWorld);
        Vector3 chassisVelocityRobot = robotRoot.InverseTransformDirection(chassisVelocityWorld);

        sensor.AddObservation(targetFromMuzzleRobot.x / positionNormalizer);
        sensor.AddObservation(targetFromMuzzleRobot.z / positionNormalizer);
        sensor.AddObservation(targetFromMuzzleRobot.y / positionNormalizer);

        sensor.AddObservation(chassisVelocityRobot.x / velocityNormalizer);
        sensor.AddObservation(chassisVelocityRobot.z / velocityNormalizer);

        sensor.AddObservation(currentYawDeg / maxYawDeg);
        sensor.AddObservation(currentPitchDeg / maxPitchDeg);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveRobotFromAToB();

        if (waitingForVisualEnd)
        {
            visualTimer += Time.fixedDeltaTime;

            if (visualTimer >= visualSecondsAfterShot)
            {
                EndEpisode();
            }

            return;
        }

        if (hasShot)
            return;

        float yawAction = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float pitchAction = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float rpmAction = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);

        currentYawDeg += yawAction * maxYawDeltaPerStep;
        currentYawDeg = Mathf.Clamp(currentYawDeg, minYawDeg, maxYawDeg);

        currentPitchDeg += pitchAction * maxPitchDeltaPerStep;
        currentPitchDeg = Mathf.Clamp(currentPitchDeg, minPitchDeg, maxPitchDeg);

        currentRpm = (rpmAction + 1f) * 0.5f * maxRpm;

        ApplyTurretPose();

        AddReward(timePenalty);

        if (StepCount >= aimStepsBeforeShot)
        {
            FireAndEvaluateShot();
            hasShot = true;
        }
    }

    private void ConfigureFieldTargetAndRobotHeading()
    {
        Vector3 cornerPosition = GetTargetCornerPosition();

        if (autoPlaceTargetAtCorner && target != null)
        {
            target.position = new Vector3(
                cornerPosition.x,
                target.position.y,
                cornerPosition.z
            );
        }

        if (autoFaceRobotTowardTargetCorner)
        {
            Vector3 fieldToTarget = cornerPosition - fieldCenter.position;
            fieldToTarget.y = 0f;

            if (fieldToTarget.sqrMagnitude > 0.001f)
            {
                robotBaseRotation = Quaternion.LookRotation(fieldToTarget.normalized, Vector3.up);
            }
            else
            {
                robotBaseRotation = robotRoot.rotation;
            }
        }
        else
        {
            robotBaseRotation = robotRoot.rotation;
        }
    }

    private Vector3 GetTargetCornerPosition()
    {
        float half = fieldSizeMeters * 0.5f;
        Vector3 center = fieldCenter.position;

        switch (targetCorner)
        {
            case TargetCorner.PositiveXPositiveZ:
                return center + new Vector3(half, 0f, half);

            case TargetCorner.NegativeXPositiveZ:
                return center + new Vector3(-half, 0f, half);

            case TargetCorner.PositiveXNegativeZ:
                return center + new Vector3(half, 0f, -half);

            case TargetCorner.NegativeXNegativeZ:
                return center + new Vector3(-half, 0f, -half);

            default:
                return center + new Vector3(half, 0f, half);
        }
    }

    private void MoveRobotFromAToB()
    {
        Vector3 flatRobot = new Vector3(robotRoot.position.x, 0f, robotRoot.position.z);
        Vector3 flatB = new Vector3(pointB.x, 0f, pointB.z);

        if (Vector3.Distance(flatRobot, flatB) <= 0.05f)
        {
            chassisVelocityWorld = Vector3.zero;
            return;
        }

        Vector3 movement = moveDirectionWorld * fixedMoveSpeed * Time.fixedDeltaTime;
        robotRoot.position += movement;

        chassisVelocityWorld = moveDirectionWorld * fixedMoveSpeed;

        if (keepRobotHeadingFixed)
        {
            robotRoot.rotation = robotBaseRotation;
        }
    }

    private void GenerateValidPointAAndB()
    {
        bool found = false;

        for (int attempt = 0; attempt < maxPointSearchAttempts; attempt++)
        {
            Vector3 candidateA = GenerateRobotPositionFromDocTargetRange();
            Vector3 candidateB = GenerateRobotPositionFromDocTargetRange();

            bool aInsideField = IsInsideField(candidateA);
            bool bInsideField = IsInsideField(candidateB);
            bool farEnough = Vector3.Distance(candidateA, candidateB) >= minPathDistance;

            Vector3 estimatedShootPoint = EstimateShootPoint(candidateA, candidateB);

            bool aReachable = !validateReachability || IsTargetReachableFromRobotPose(candidateA);
            bool bReachable = !validateReachability || IsTargetReachableFromRobotPose(candidateB);
            bool shootReachable = !validateReachability || IsTargetReachableFromRobotPose(estimatedShootPoint);

            Vector3 targetAtShoot = GetTargetRelativeFromRobotPose(estimatedShootPoint);
            bool yawActuallyNeeded = Mathf.Abs(targetAtShoot.x) >= minAbsSideOffsetAtShoot;

            Vector3 candidateDirection = candidateB - candidateA;
            candidateDirection.y = 0f;

            if (candidateDirection.sqrMagnitude < 0.001f)
                continue;

            Vector3 candidateVelocityWorld = candidateDirection.normalized * fixedMoveSpeed;
            Vector3 candidateVelocityRobot = Quaternion.Inverse(robotBaseRotation) * candidateVelocityWorld;

            bool hasLateralMotion =
                Mathf.Abs(candidateVelocityRobot.x) >= minLateralVelocityForYawTraining;

            if (
                aInsideField &&
                bInsideField &&
                farEnough &&
                aReachable &&
                bReachable &&
                shootReachable &&
                yawActuallyNeeded &&
                hasLateralMotion
            )
            {
                pointA = candidateA;
                pointB = candidateB;
                found = true;
                break;
            }
        }

        if (!found)
        {
            Debug.LogWarning(
                "Could not find valid A/B points with current constraints. " +
                "Lower Min Abs Side Offset At Shoot / Min Lateral Velocity / Min Path Distance, " +
                "or check field center, target corner, and robot heading."
            );

            pointA = GenerateRobotPositionFromDocTargetRange();
            pointB = GenerateRobotPositionFromDocTargetRange();
        }
    }

    private Vector3 GenerateRobotPositionFromDocTargetRange()
    {
        float dx = Random.Range(-maxTargetSide, maxTargetSide);
        float dy = Random.Range(minTargetForward, maxTargetForward);

        Vector3 robotRight = robotBaseRotation * Vector3.right;
        Vector3 robotForward = robotBaseRotation * Vector3.forward;

        Vector3 oldPosition = robotRoot.position;
        Quaternion oldRotation = robotRoot.rotation;

        robotRoot.rotation = robotBaseRotation;

        Vector3 muzzleLocalOffset = robotRoot.InverseTransformPoint(muzzle.position);

        robotRoot.position = oldPosition;
        robotRoot.rotation = oldRotation;

        Vector3 muzzlePlanarOffsetWorld =
            robotRight * muzzleLocalOffset.x +
            robotForward * muzzleLocalOffset.z;

        Vector3 targetFlat = new Vector3(
            target.position.x,
            robotRoot.position.y,
            target.position.z
        );

        Vector3 robotPosition =
            targetFlat
            - robotRight * dx
            - robotForward * dy
            - muzzlePlanarOffsetWorld;

        return robotPosition;
    }

    private Vector3 EstimateShootPoint(Vector3 candidateA, Vector3 candidateB)
    {
        Vector3 path = candidateB - candidateA;
        path.y = 0f;

        float totalDistance = path.magnitude;

        if (totalDistance <= 0.001f)
            return candidateA;

        float travelBeforeShot = fixedMoveSpeed * Time.fixedDeltaTime * aimStepsBeforeShot;
        float progress = Mathf.Clamp01(travelBeforeShot / totalDistance);

        return Vector3.Lerp(candidateA, candidateB, progress);
    }

    private Vector3 GetTargetRelativeFromRobotPose(Vector3 robotPosition)
    {
        Vector3 oldPosition = robotRoot.position;
        Quaternion oldRotation = robotRoot.rotation;

        robotRoot.position = robotPosition;

        if (keepRobotHeadingFixed)
        {
            robotRoot.rotation = robotBaseRotation;
        }

        ApplyTurretPose();

        Vector3 targetFromMuzzleWorld = target.position - muzzle.position;
        Vector3 targetFromMuzzleRobot = robotRoot.InverseTransformDirection(targetFromMuzzleWorld);

        robotRoot.position = oldPosition;
        robotRoot.rotation = oldRotation;

        return targetFromMuzzleRobot;
    }

    private bool IsTargetReachableFromRobotPose(Vector3 robotPosition)
    {
        Vector3 targetFromMuzzleRobot = GetTargetRelativeFromRobotPose(robotPosition);

        float dx = targetFromMuzzleRobot.x;
        float dy = targetFromMuzzleRobot.z;
        float dz = targetFromMuzzleRobot.y;

        float requiredYaw = Mathf.Atan2(dx, dy) * Mathf.Rad2Deg;

        float horizontalDistance = new Vector2(dx, dy).magnitude;
        float lineOfSightPitch = Mathf.Atan2(dz, horizontalDistance) * Mathf.Rad2Deg;

        bool targetInFront = dy >= minActualForwardAtShoot && dy <= maxActualForwardAtShoot;
        bool targetSideValid = Mathf.Abs(dx) <= maxActualSideAtShoot;
        bool yawValid = requiredYaw >= minYawDeg && requiredYaw <= maxYawDeg;
        bool pitchValid = lineOfSightPitch >= minPitchDeg && lineOfSightPitch <= maxPitchDeg;

        return targetInFront && targetSideValid && yawValid && pitchValid;
    }

    private bool IsInsideField(Vector3 position)
    {
        float half = fieldSizeMeters * 0.5f;
        Vector3 local = fieldCenter.InverseTransformPoint(position);

        return Mathf.Abs(local.x) <= half && Mathf.Abs(local.z) <= half;
    }

    private void ApplyTurretPose()
    {
        Quaternion yawDelta =
            Quaternion.AngleAxis(currentYawDeg * yawDirection, yawLocalAxis.normalized);

        Quaternion pitchDelta =
            Quaternion.AngleAxis(currentPitchDeg * pitchDirection, pitchLocalAxis.normalized);

        turretYawPivot.localRotation = yawBaseRotation * yawDelta;
        hoodPitchPivot.localRotation = pitchBaseRotation * pitchDelta;
    }

    private void FireAndEvaluateShot()
    {
        bool hit = EvaluateShotMath(out float missDistance, out Vector3 projectileVelocity);

        if (drawDebugTrajectory)
        {
            DrawTrajectoryDebug(muzzle.position, projectileVelocity);
        }

        if (spawnVisualBall)
        {
            ShootVisualBall(projectileVelocity);
        }

        if (hit)
        {
            RecordEvaluationResult(true, 0f);
            SetReward(hitReward);
        }
        else
        {
            RecordEvaluationResult(false, missDistance);
            AddReward(-missDistance * missPenaltyMultiplier);
        }

        if (spawnVisualBall && delayEpisodeForVisuals)
        {
            waitingForVisualEnd = true;
            visualTimer = 0f;
        }
        else
        {
            EndEpisode();
        }
    }

    private bool EvaluateShotMath(out float missDistance, out Vector3 projectileVelocity)
    {
        Vector3 start = muzzle.position;
        Vector3 targetPoint = target.position;

        projectileVelocity = CalculateProjectileVelocity();

        Vector3 relativeTarget = targetPoint - start;

        float targetHeight = relativeTarget.y;
        float verticalVelocity = projectileVelocity.y;
        float gravity = Mathf.Abs(Physics.gravity.y);

        float a = 0.5f * gravity;
        float b = -verticalVelocity;
        float c = targetHeight;

        float discriminant = b * b - 4f * a * c;

        missDistance = 10f;

        if (discriminant < 0f)
        {
            float maxHeightReached =
                (verticalVelocity * verticalVelocity) / (2f * gravity);

            missDistance =
                5f + Mathf.Abs(targetHeight - maxHeightReached);

            return false;
        }

        float sqrtD = Mathf.Sqrt(discriminant);

        float t1 = (-b - sqrtD) / (2f * a);
        float t2 = (-b + sqrtD) / (2f * a);

        float bestMiss = float.MaxValue;

        if (t1 > 0f)
        {
            Vector3 positionAtT =
                start + projectileVelocity * t1 + 0.5f * Physics.gravity * t1 * t1;

            float miss = HorizontalMissDistance(positionAtT, targetPoint);
            bestMiss = Mathf.Min(bestMiss, miss);
        }

        if (t2 > 0f)
        {
            Vector3 positionAtT =
                start + projectileVelocity * t2 + 0.5f * Physics.gravity * t2 * t2;

            float miss = HorizontalMissDistance(positionAtT, targetPoint);
            bestMiss = Mathf.Min(bestMiss, miss);
        }

        if (bestMiss == float.MaxValue)
        {
            missDistance = 10f;
            return false;
        }

        missDistance = bestMiss;
        return missDistance <= hitRadius;
    }

    private Vector3 CalculateProjectileVelocity()
    {
        float wheelRadiusMeters = 0.0508f;

        float exitSpeed =
            currentRpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        Vector3 shooterVelocity = muzzle.forward * exitSpeed;

        return shooterVelocity + chassisVelocityWorld;
    }

    private float HorizontalMissDistance(Vector3 projectilePosition, Vector3 targetPoint)
    {
        Vector2 projectileXZ = new Vector2(projectilePosition.x, projectilePosition.z);
        Vector2 targetXZ = new Vector2(targetPoint.x, targetPoint.z);

        return Vector2.Distance(projectileXZ, targetXZ);
    }

    private void ShootVisualBall(Vector3 projectileVelocity)
    {
        if (ballPrefab == null)
            return;

        Rigidbody ball = Instantiate(ballPrefab, muzzle.position, Quaternion.identity);
        activeBall = ball;

        ball.linearVelocity = projectileVelocity;
    }

    private void DrawTrajectoryDebug(Vector3 start, Vector3 projectileVelocity)
    {
        Vector3 previous = start;

        for (int i = 1; i <= debugTrajectorySegments; i++)
        {
            float t = debugTrajectorySeconds * i / debugTrajectorySegments;

            Vector3 point =
                start + projectileVelocity * t + 0.5f * Physics.gravity * t * t;

            Debug.DrawLine(previous, point, Color.magenta, debugTrajectoryDuration);
            previous = point;
        }
    }

    private void CleanupBall()
    {
        if (activeBall != null)
        {
            Destroy(activeBall.gameObject);
            activeBall = null;
        }
    }

    private void ValidateTargetHeightOnce()
    {
        if (warnedTargetHeight)
            return;

        float actualDz = target.position.y - muzzle.position.y;
        float error = Mathf.Abs(actualDz - targetRelativeHeight);

        if (error > 0.25f)
        {
            Debug.LogWarning(
                $"Target relative height ΔZ is {actualDz:F2} m, but document value is {targetRelativeHeight:F2} m. " +
                "Adjust target height or muzzle height if you want strict doc matching."
            );
        }

        warnedTargetHeight = true;
    }

    private void RecordEvaluationResult(bool hit, float missDistance)
    {
        if (!evaluationMode)
            return;

        if (!evaluationStarted)
        {
            evaluationStarted = true;
            evalStartTime = Time.realtimeSinceStartup;
        }

        evalEpisodesCompleted++;

        if (hit)
        {
            evalHits++;
        }
        else
        {
            evalMisses++;
            evalTotalMissDistance += missDistance;
        }

        float accuracy = 100f * evalHits / Mathf.Max(1, evalEpisodesCompleted);
        float averageMissDistance = evalMisses > 0 ? evalTotalMissDistance / evalMisses : 0f;
        float elapsed = Time.realtimeSinceStartup - evalStartTime;

        if (logEveryEpisode)
        {
            Debug.Log(
                $"EVAL {evalEpisodesCompleted}/{evaluationEpisodeLimit} | " +
                $"Hits: {evalHits} | Misses: {evalMisses} | " +
                $"Accuracy: {accuracy:F2}% | Avg Miss: {averageMissDistance:F3} m | " +
                $"Elapsed: {elapsed:F1}s"
            );
        }

        if (evalEpisodesCompleted >= evaluationEpisodeLimit)
        {
            Debug.Log(
                $"FINAL EVALUATION RESULT | Episodes: {evalEpisodesCompleted} | " +
                $"Hits: {evalHits} | Misses: {evalMisses} | " +
                $"Accuracy: {accuracy:F2}% | Avg Miss: {averageMissDistance:F3} m | " +
                $"Elapsed: {elapsed:F1}s"
            );

            if (stopAfterEvaluation)
            {
                Time.timeScale = 0f;
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> actions = actionsOut.ContinuousActions;

        float yaw = 0f;
        float pitch = 0f;
        float rpm = -1f;

        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            yaw = -1f;

        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            yaw = 1f;

        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
            pitch = 1f;

        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            pitch = -1f;

        if (Input.GetKey(KeyCode.Space))
            rpm = 1f;

        actions[0] = yaw;
        actions[1] = pitch;
        actions[2] = rpm;
    }

    private void OnDrawGizmosSelected()
    {
        if (fieldCenter == null)
            return;

        Gizmos.color = Color.green;
        DrawFieldGizmo();

        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(pointA, 0.08f);

        Gizmos.color = Color.red;
        Gizmos.DrawSphere(pointB, 0.08f);

        Gizmos.color = Color.cyan;
        Gizmos.DrawLine(pointA, pointB);

        Vector3 shootPoint = EstimateShootPoint(pointA, pointB);
        Gizmos.color = Color.magenta;
        Gizmos.DrawSphere(shootPoint, 0.1f);
    }

    private void DrawFieldGizmo()
    {
        float half = fieldSizeMeters * 0.5f;
        Vector3 center = fieldCenter.position;

        Vector3 p1 = center + new Vector3(-half, 0f, -half);
        Vector3 p2 = center + new Vector3(-half, 0f, half);
        Vector3 p3 = center + new Vector3(half, 0f, half);
        Vector3 p4 = center + new Vector3(half, 0f, -half);

        Gizmos.DrawLine(p1, p2);
        Gizmos.DrawLine(p2, p3);
        Gizmos.DrawLine(p3, p4);
        Gizmos.DrawLine(p4, p1);
    }
}
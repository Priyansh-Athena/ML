using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class SevenParameterMovingShooterAgent : Agent
{
    [Header("Scene References")]
    public Transform circleCenter;
    public Transform robotRoot;
    public Transform turretYawPivot;
    public Transform hoodPitchPivot;
    public Transform muzzle;
    public Transform target;
    public Rigidbody ballPrefab;

    [Header("Document Target Ranges")]
    public float minTargetForward = 0.5f;     // ΔY min from doc
    public float maxTargetForward = 3.5f;     // ΔY max from doc
    public float maxTargetSide = 1.5f;        // ΔX range = -1.5 to +1.5
    public float targetRelativeHeight = 1.2f; // ΔZ from doc

    [Header("Arena")]
    public float circleRadius = 5f;

    [Header("Robot Motion A -> B")]
    public float fixedMoveSpeed = 1.5f;
    public float minPathDistance = 1.75f;
    public bool keepRobotHeadingFixed = true;

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
    public bool spawnVisualBall = false;

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
    private Rigidbody activeBall;

    private void Awake()
    {
        robotBaseRotation = robotRoot.rotation;
        yawBaseRotation = turretYawPivot.localRotation;
        pitchBaseRotation = hoodPitchPivot.localRotation;
    }

    public override void OnEpisodeBegin()
    {
        if (evaluationMode && !evaluationStarted)
        {
            evaluationStarted = true;
            evalStartTime = Time.realtimeSinceStartup;
            Time.timeScale = 1f;
        }

        CleanupBall();

        hasShot = false;

        currentYawDeg = 0f;
        currentPitchDeg = 45f;
        currentRpm = maxRpm * 0.7f;

        if (keepRobotHeadingFixed)
        {
            robotRoot.rotation = robotBaseRotation;
        }

        GenerateValidPointAAndB();

        robotRoot.position = pointA;

        Vector3 flatDirection = pointB - pointA;
        flatDirection.y = 0f;

        moveDirectionWorld = flatDirection.normalized;
        chassisVelocityWorld = moveDirectionWorld * fixedMoveSpeed;

        ApplyTurretPose();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Exactly 7 observations from the document:
        // ΔX, ΔY, ΔZ, Vx, Vy, current yaw, current pitch.

        Vector3 targetFromMuzzleWorld = target.position - muzzle.position;
        Vector3 targetFromMuzzleRobot = robotRoot.InverseTransformDirection(targetFromMuzzleWorld);

        Vector3 chassisVelocityRobot = robotRoot.InverseTransformDirection(chassisVelocityWorld);

        // 1. ΔX: target left/right relative to robot
        sensor.AddObservation(targetFromMuzzleRobot.x / positionNormalizer);

        // 2. ΔY: target forward distance relative to robot
        // Unity local forward is Z.
        sensor.AddObservation(targetFromMuzzleRobot.z / positionNormalizer);

        // 3. ΔZ: target height relative to muzzle
        // Unity height is Y.
        sensor.AddObservation(targetFromMuzzleRobot.y / positionNormalizer);

        // 4. Vx: robot-local sideways velocity
        sensor.AddObservation(chassisVelocityRobot.x / velocityNormalizer);

        // 5. Vy: robot-local forward velocity
        sensor.AddObservation(chassisVelocityRobot.z / velocityNormalizer);

        // 6. Current yaw normalized to approximately [-1, +1]
        sensor.AddObservation(currentYawDeg / maxYawDeg);

        // 7. Current pitch normalized to [0, 1]
        sensor.AddObservation(currentPitchDeg / maxPitchDeg);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveRobotFromAToB();

        if (!hasShot)
        {
            float yawAction = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float pitchAction = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            float rpmAction = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);

            // Document: yaw change = action[0] * 5 degrees
            currentYawDeg += yawAction * maxYawDeltaPerStep;
            currentYawDeg = Mathf.Clamp(currentYawDeg, minYawDeg, maxYawDeg);

            // Document: pitch change = action[1] * 2 degrees, clipped 0 to 90
            currentPitchDeg += pitchAction * maxPitchDeltaPerStep;
            currentPitchDeg = Mathf.Clamp(currentPitchDeg, minPitchDeg, maxPitchDeg);

            // Document: RPM maps from [-1, 1] to [0, 6000]
            currentRpm = (rpmAction + 1f) * 0.5f * maxRpm;

            ApplyTurretPose();

            AddReward(timePenalty);

            if (StepCount >= aimStepsBeforeShot)
            {
                FireAndEvaluateShot();
                hasShot = true;
            }
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

        for (int attempt = 0; attempt < 500; attempt++)
        {
            Vector3 candidateA = GenerateRobotPositionFromDocTargetRange();
            Vector3 candidateB = GenerateRobotPositionFromDocTargetRange();

            bool aInsideCircle = IsInsideCircle(candidateA);
            bool bInsideCircle = IsInsideCircle(candidateB);
            bool farEnough = Vector3.Distance(candidateA, candidateB) >= minPathDistance;

            if (aInsideCircle && bInsideCircle && farEnough)
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
                "Could not find valid A/B points inside the circle using document ranges. " +
                "Increase circle radius or move the fixed target closer to the circle."
            );

            pointA = GenerateRobotPositionFromDocTargetRange();
            pointB = GenerateRobotPositionFromDocTargetRange();
        }
    }

    private Vector3 GenerateRobotPositionFromDocTargetRange()
    {
        // The document defines the target relative to the robot:
        // ΔX in [-1.5, 1.5]
        // ΔY in [0.5, 3.5]
        // ΔZ = 1.2
        //
        // Since target is fixed in Unity, we reverse this:
        // robot position = target position - ΔX * robotRight - ΔY * robotForward

        float dx = Random.Range(-maxTargetSide, maxTargetSide);
        float dy = Random.Range(minTargetForward, maxTargetForward);

        Vector3 robotRight = robotBaseRotation * Vector3.right;
        Vector3 robotForward = robotBaseRotation * Vector3.forward;

        Vector3 targetFlat = new Vector3(target.position.x, robotRoot.position.y, target.position.z);

        Vector3 robotPosition =
            targetFlat
            - robotRight * dx
            - robotForward * dy;

        return robotPosition;
    }

    private bool IsInsideCircle(Vector3 position)
    {
        Vector3 flatOffset = new Vector3(
            position.x - circleCenter.position.x,
            0f,
            position.z - circleCenter.position.z
        );

        return flatOffset.magnitude <= circleRadius;
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
        bool hit = EvaluateShotMath(out float missDistance);

        if (spawnVisualBall)
        {
            ShootVisualBall();
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

        EndEpisode();
    }

    private bool EvaluateShotMath(out float missDistance)
    {
        Vector3 start = muzzle.position;
        Vector3 targetPoint = target.position;

        float wheelRadiusMeters = 0.0508f;

        float exitSpeed =
            currentRpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        Vector3 shooterVelocity = muzzle.forward * exitSpeed;

        // This is the inherited momentum from the moving chassis.
        Vector3 projectileVelocity = shooterVelocity + chassisVelocityWorld;

        Vector3 relativeTarget = targetPoint - start;

        float targetHeight = relativeTarget.y;
        float verticalVelocity = projectileVelocity.y;
        float gravity = Mathf.Abs(Physics.gravity.y);

        // targetHeight = verticalVelocity * t - 0.5 * gravity * t^2
        // 0.5*g*t^2 - verticalVelocity*t + targetHeight = 0

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

    private float HorizontalMissDistance(Vector3 projectilePosition, Vector3 targetPoint)
    {
        Vector2 projectileXZ = new Vector2(projectilePosition.x, projectilePosition.z);
        Vector2 targetXZ = new Vector2(targetPoint.x, targetPoint.z);

        return Vector2.Distance(projectileXZ, targetXZ);
    }

    private void ShootVisualBall()
    {
        if (ballPrefab == null)
            return;

        Rigidbody ball = Instantiate(ballPrefab, muzzle.position, Quaternion.identity);
        activeBall = ball;

        float wheelRadiusMeters = 0.0508f;

        float exitSpeed =
            currentRpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        Vector3 shooterVelocity = muzzle.forward * exitSpeed;
        Vector3 projectileVelocity = shooterVelocity + chassisVelocityWorld;

        ball.linearVelocity = projectileVelocity;
    }

    private void CleanupBall()
    {
        if (activeBall != null)
        {
            Destroy(activeBall.gameObject);
            activeBall = null;
        }
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
        if (circleCenter == null)
            return;

        Gizmos.color = Color.green;
        DrawCircleGizmo(circleCenter.position, circleRadius);

        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(pointA, 0.08f);

        Gizmos.color = Color.red;
        Gizmos.DrawSphere(pointB, 0.08f);

        Gizmos.color = Color.cyan;
        Gizmos.DrawLine(pointA, pointB);
    }

    private void DrawCircleGizmo(Vector3 center, float radius)
    {
        int segments = 96;
        Vector3 previous = center + new Vector3(radius, 0f, 0f);

        for (int i = 1; i <= segments; i++)
        {
            float angle = i * Mathf.PI * 2f / segments;

            Vector3 next = center + new Vector3(
                Mathf.Cos(angle) * radius,
                0f,
                Mathf.Sin(angle) * radius
            );

            Gizmos.DrawLine(previous, next);
            previous = next;
        }
    }
}
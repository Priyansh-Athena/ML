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

    [Header("Arena")]
    public float circleRadius = 5f;

    [Header("Robot Motion A -> B")]
    public float fixedMoveSpeed = 1.5f;
    public float arriveDistance = 0.15f;
    public bool rotateRobotAlongPath = true;

    [Header("Turret Axis Settings")]
    public Vector3 yawLocalAxis = Vector3.up;
    public Vector3 pitchLocalAxis = Vector3.up;
    public float yawDirection = 1f;
    public float pitchDirection = -1f;

    [Header("Mechanical Limits")]
    public float maxYawDeg = 360f;
    public float maxPitchDeg = 180f;
    public float maxYawDeltaPerStep = 5f;
    public float maxPitchDeltaPerStep = 2f;
    public float maxRpm = 6000f;

    [Header("Shooting")]
    public int aimStepsBeforeShot = 20;
    public float maxBallFlightTime = 2.5f;
    public float hitReward = 100f;
    public float timePenalty = -0.05f;
    public float missPenaltyMultiplier = 1f;

    [Header("Observation Normalization")]
    public float positionNormalizer = 5f;
    public float velocityNormalizer = 2f;

    private Vector3 pointA;
    private Vector3 pointB;
    private Vector3 moveDirectionWorld;
    private Vector3 chassisVelocityWorld;

    private float currentYawDeg;
    private float currentPitchDeg;
    private float currentRpm;

    private Quaternion yawBaseRotation;
    private Quaternion pitchBaseRotation;

    private bool hasShot;
    private bool hitReported;
    private float ballTimer;
    private Rigidbody activeBall;

    private void Awake()
    {
        yawBaseRotation = turretYawPivot.localRotation;
        pitchBaseRotation = hoodPitchPivot.localRotation;
    }

    public override void OnEpisodeBegin()
    {
        CleanupBall();

        hasShot = false;
        hitReported = false;
        ballTimer = 0f;

        currentYawDeg = 0f;
        currentPitchDeg = 45f;
        currentRpm = maxRpm * 0.7f;

        pointA = RandomPointInsideCircle();
        pointB = RandomPointInsideCircle();

        int safety = 0;
        while (Vector3.Distance(pointA, pointB) < circleRadius * 0.4f && safety < 50)
        {
            pointB = RandomPointInsideCircle();
            safety++;
        }

        robotRoot.position = pointA;

        Vector3 flatDirection = pointB - pointA;
        flatDirection.y = 0f;
        moveDirectionWorld = flatDirection.normalized;

        chassisVelocityWorld = moveDirectionWorld * fixedMoveSpeed;

        if (rotateRobotAlongPath && moveDirectionWorld.sqrMagnitude > 0.001f)
        {
            robotRoot.rotation = Quaternion.LookRotation(moveDirectionWorld, Vector3.up);
        }

        ApplyTurretPose();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 7 observation parameters from the document:
        // ΔX, ΔY, ΔZ, Vx, Vy, current yaw, current pitch.

        Vector3 targetFromMuzzleWorld = target.position - muzzle.position;
        Vector3 targetFromMuzzleRobot = robotRoot.InverseTransformDirection(targetFromMuzzleWorld);

        Vector3 chassisVelocityRobot = robotRoot.InverseTransformDirection(chassisVelocityWorld);

        // 1. ΔX: target left/right relative to robot/muzzle
        sensor.AddObservation(targetFromMuzzleRobot.x / positionNormalizer);

        // 2. ΔY: target forward distance.
        // In Unity, forward is local Z.
        sensor.AddObservation(targetFromMuzzleRobot.z / positionNormalizer);

        // 3. ΔZ: target height.
        // In Unity, height is local Y.
        sensor.AddObservation(targetFromMuzzleRobot.y / positionNormalizer);

        // 4. Vx: robot-local sideways velocity
        sensor.AddObservation(chassisVelocityRobot.x / velocityNormalizer);

        // 5. Vy: robot-local forward velocity
        sensor.AddObservation(chassisVelocityRobot.z / velocityNormalizer);

        // 6. current yaw, normalized 0 to 1
        sensor.AddObservation(currentYawDeg / maxYawDeg);

        // 7. current pitch, normalized 0 to 1
        sensor.AddObservation(currentPitchDeg / maxPitchDeg);

        if (StepCount % 20 == 0)
        {
            Debug.Log("SevenParameterMovingShooterAgent sent 7 observations.");
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveRobotFromAToB();

        if (!hasShot)
        {
            float yawAction = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float pitchAction = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            float rpmAction = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);

            currentYawDeg += yawAction * maxYawDeltaPerStep;

            // Yaw is 0 to 360, so wrap it instead of clamping it.
            currentYawDeg = Mathf.Repeat(currentYawDeg, maxYawDeg);

            currentPitchDeg += pitchAction * maxPitchDeltaPerStep;

            // Pitch is 0 to 180.
            currentPitchDeg = Mathf.Clamp(currentPitchDeg, 0f, maxPitchDeg);

            // RPM action maps from [-1, 1] to [0, maxRpm].
            currentRpm = (rpmAction + 1f) * 0.5f * maxRpm;

            ApplyTurretPose();

            AddReward(timePenalty);

            if (StepCount >= aimStepsBeforeShot)
            {
                ShootBall();
                hasShot = true;
            }
        }
        else
        {
            ballTimer += Time.fixedDeltaTime;

            if (hitReported)
            {
                SetReward(hitReward);
                EndEpisode();
                return;
            }

            if (ballTimer >= maxBallFlightTime)
            {
                float missDistance = GetMissDistance();
                AddReward(-missDistance * missPenaltyMultiplier);
                EndEpisode();
                return;
            }
        }

        if (HasReachedPointB() && !hasShot)
        {
            ShootBall();
            hasShot = true;
        }
    }

    private void MoveRobotFromAToB()
    {
        if (HasReachedPointB())
        {
            chassisVelocityWorld = Vector3.zero;
            return;
        }

        Vector3 movement = moveDirectionWorld * fixedMoveSpeed * Time.fixedDeltaTime;
        robotRoot.position += movement;

        chassisVelocityWorld = moveDirectionWorld * fixedMoveSpeed;

        if (rotateRobotAlongPath && moveDirectionWorld.sqrMagnitude > 0.001f)
        {
            robotRoot.rotation = Quaternion.LookRotation(moveDirectionWorld, Vector3.up);
        }
    }

    private bool HasReachedPointB()
    {
        Vector3 flatRobot = new Vector3(robotRoot.position.x, 0f, robotRoot.position.z);
        Vector3 flatB = new Vector3(pointB.x, 0f, pointB.z);

        return Vector3.Distance(flatRobot, flatB) <= arriveDistance;
    }

    private Vector3 RandomPointInsideCircle()
    {
        float randomRadius = circleRadius * Mathf.Sqrt(Random.Range(0f, 1f));
        float randomAngle = Random.Range(0f, Mathf.PI * 2f);

        float x = Mathf.Cos(randomAngle) * randomRadius;
        float z = Mathf.Sin(randomAngle) * randomRadius;

        Vector3 center = circleCenter.position;

        return new Vector3(
            center.x + x,
            robotRoot.position.y,
            center.z + z
        );
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

    private void ShootBall()
    {
        if (ballPrefab == null)
        {
            Debug.LogError("Ball Prefab is not assigned.");
            return;
        }

        Rigidbody ball = Instantiate(ballPrefab, muzzle.position, Quaternion.identity);
        activeBall = ball;

        ProjectileHitReporter reporter = ball.GetComponent<ProjectileHitReporter>();

        if (reporter != null)
        {
            reporter.Initialize(this);
        }

        float wheelRadiusMeters = 0.0508f;

        float exitSpeed =
            currentRpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        // You said muzzle.forward is the ball direction.
        Vector3 shooterVelocity = muzzle.forward * exitSpeed;

        // Projectile inherits robot chassis velocity.
        Vector3 inheritedRobotVelocity = chassisVelocityWorld;

        ball.linearVelocity = shooterVelocity + inheritedRobotVelocity;
    }

    public void ReportProjectileHit()
    {
        hitReported = true;
    }

    private float GetMissDistance()
    {
        if (activeBall != null)
        {
            return Vector3.Distance(activeBall.position, target.position);
        }

        return Vector3.Distance(muzzle.position, target.position);
    }

    private void CleanupBall()
    {
        if (activeBall != null)
        {
            Destroy(activeBall.gameObject);
            activeBall = null;
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
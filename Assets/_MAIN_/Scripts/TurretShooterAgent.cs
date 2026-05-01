using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class TurretShooterAgent : Agent
{
    [Header("Scene References")]
    public Transform circleCenter;
    public Transform robotRoot;
    public Transform turretYawPivot;
    public Transform hoodPitchPivot;
    public Transform muzzle;
    public Transform target;
    public Rigidbody ballPrefab;

    [Header("Circle / Arena")]
    public float circleRadius = 3f;
    public float targetMaxRadiusMultiplier = 1.5f;
    public float targetHeight = 1.2f;

    [Header("Robot Movement")]
    public float maxChassisSpeed = 2f;
    public bool rotateRobotInMoveDirection = true;

    [Header("Manual Debug")]
    public bool manualKeyboardDebug = false;

    [Header("Rotation Axis Settings")]
    public Vector3 yawLocalAxis = Vector3.up;
    public Vector3 pitchLocalAxis = Vector3.up;
    public float yawDirection = 1f;
    public float pitchDirection = -1f;

    [Header("Mechanical Limits")]
    public float maxYawDeg = 90f;
    public float maxPitchDeg = 90f;
    public float maxYawDeltaPerStep = 5f;
    public float maxPitchDeltaPerStep = 2f;
    public float maxRpm = 6000f;

    [Header("Shot Timing")]
    public int aimStepsBeforeShot = 20;
    public float maxBallFlightTime = 2.5f;

    [Header("Observation Normalization")]
    public float positionNormalizer = 5f;
    public float velocityNormalizer = 2f;

    private float currentYawDeg;
    private float currentPitchDeg;
    private float currentRpm;

    private Vector2 chassisVelocityXZ;

    private Quaternion yawBaseRotation;
    private Quaternion pitchBaseRotation;

    private bool hasShot;
    private bool hitReported;
    private float shotTimer;
    private Rigidbody activeBall;

    private void Awake()
    {
        yawBaseRotation = turretYawPivot.localRotation;
        pitchBaseRotation = hoodPitchPivot.localRotation;
    }

    private void Update()
    {
        if (!manualKeyboardDebug)
            return;

        float yawInput = 0f;
        float pitchInput = 0f;

        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            yawInput = -1f;

        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            yawInput = 1f;

        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
            pitchInput = 1f;

        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            pitchInput = -1f;

        currentYawDeg += yawInput * maxYawDeltaPerStep * Time.deltaTime * 20f;
        currentPitchDeg += pitchInput * maxPitchDeltaPerStep * Time.deltaTime * 20f;

        currentYawDeg = Mathf.Clamp(currentYawDeg, -maxYawDeg, maxYawDeg);
        currentPitchDeg = Mathf.Clamp(currentPitchDeg, 0f, maxPitchDeg);

        ApplyTurretPose();
    }

    public override void OnEpisodeBegin()
    {
        CleanupBall();

        hasShot = false;
        hitReported = false;
        shotTimer = 0f;

        currentYawDeg = 0f;
        currentPitchDeg = 45f;
        currentRpm = maxRpm * 0.7f;

        RandomizeRobotPositionInsideCircle();
        RandomizeRobotVelocity();
        RandomizeTargetPosition();

        ApplyTurretPose();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 targetFromMuzzleWorld = target.position - muzzle.position;
        Vector3 targetFromMuzzleRobot = robotRoot.InverseTransformDirection(targetFromMuzzleWorld);

        Vector3 robotOffsetFromCenter = robotRoot.position - circleCenter.position;

        sensor.AddObservation(targetFromMuzzleRobot.x / positionNormalizer);
        sensor.AddObservation(targetFromMuzzleRobot.y / positionNormalizer);
        sensor.AddObservation(targetFromMuzzleRobot.z / positionNormalizer);

        sensor.AddObservation(chassisVelocityXZ.x / velocityNormalizer);
        sensor.AddObservation(chassisVelocityXZ.y / velocityNormalizer);

        sensor.AddObservation(robotOffsetFromCenter.x / circleRadius);
        sensor.AddObservation(robotOffsetFromCenter.z / circleRadius);

        sensor.AddObservation(currentYawDeg / maxYawDeg);
        sensor.AddObservation(currentPitchDeg / maxPitchDeg);
        sensor.AddObservation(currentRpm / maxRpm);

        sensor.AddObservation(hasShot ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveRobotInsideCircle();

        if (manualKeyboardDebug)
            return;

        if (!hasShot)
        {
            float yawAction = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float pitchAction = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            float rpmAction = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);

            currentYawDeg += yawAction * maxYawDeltaPerStep;
            currentPitchDeg += pitchAction * maxPitchDeltaPerStep;

            currentYawDeg = Mathf.Clamp(currentYawDeg, -maxYawDeg, maxYawDeg);
            currentPitchDeg = Mathf.Clamp(currentPitchDeg, 0f, maxPitchDeg);

            currentRpm = (rpmAction + 1f) * 0.5f * maxRpm;

            ApplyTurretPose();

            AddReward(-0.01f);

            if (StepCount >= aimStepsBeforeShot)
            {
                ShootVisibleBall();
                hasShot = true;
            }
        }
        else
        {
            shotTimer += Time.fixedDeltaTime;

            if (hitReported)
            {
                SetReward(100f);
                EndEpisode();
                return;
            }

            if (shotTimer >= maxBallFlightTime)
            {
                float missDistance = Vector3.Distance(activeBall != null ? activeBall.position : muzzle.position, target.position);
                AddReward(-missDistance);
                EndEpisode();
            }
        }
    }

    private void RandomizeRobotPositionInsideCircle()
    {
        float randomRadius = circleRadius * Mathf.Sqrt(Random.Range(0f, 1f));
        float randomAngle = Random.Range(0f, Mathf.PI * 2f);

        float x = Mathf.Cos(randomAngle) * randomRadius;
        float z = Mathf.Sin(randomAngle) * randomRadius;

        Vector3 center = circleCenter.position;

        robotRoot.position = new Vector3(
            center.x + x,
            robotRoot.position.y,
            center.z + z
        );
    }

    private void RandomizeRobotVelocity()
    {
        float speed = Random.Range(0.5f, maxChassisSpeed);
        float angle = Random.Range(0f, Mathf.PI * 2f);

        chassisVelocityXZ = new Vector2(
            Mathf.Cos(angle) * speed,
            Mathf.Sin(angle) * speed
        );
    }

    private void RandomizeTargetPosition()
    {
        float maxTargetRadius = circleRadius * targetMaxRadiusMultiplier;

        float randomRadius = maxTargetRadius * Mathf.Sqrt(Random.Range(0f, 1f));
        float randomAngle = Random.Range(0f, Mathf.PI * 2f);

        float x = Mathf.Cos(randomAngle) * randomRadius;
        float z = Mathf.Sin(randomAngle) * randomRadius;

        Vector3 center = circleCenter.position;

        target.position = new Vector3(
            center.x + x,
            targetHeight,
            center.z + z
        );
    }

    private void MoveRobotInsideCircle()
    {
        Vector3 delta = new Vector3(
            chassisVelocityXZ.x,
            0f,
            chassisVelocityXZ.y
        ) * Time.fixedDeltaTime;

        robotRoot.position += delta;

        Vector3 center = circleCenter.position;
        Vector3 offset = robotRoot.position - center;
        Vector2 flatOffset = new Vector2(offset.x, offset.z);

        if (flatOffset.magnitude > circleRadius)
        {
            Vector2 normal = flatOffset.normalized;

            Vector3 clampedPosition = new Vector3(
                center.x + normal.x * circleRadius,
                robotRoot.position.y,
                center.z + normal.y * circleRadius
            );

            robotRoot.position = clampedPosition;

            Vector2 velocity = chassisVelocityXZ;
            Vector2 reflected = velocity - 2f * Vector2.Dot(velocity, normal) * normal;
            chassisVelocityXZ = reflected;
        }

        if (rotateRobotInMoveDirection && chassisVelocityXZ.sqrMagnitude > 0.01f)
        {
            Vector3 lookDirection = new Vector3(chassisVelocityXZ.x, 0f, chassisVelocityXZ.y);
            robotRoot.rotation = Quaternion.LookRotation(lookDirection, Vector3.up);
        }
    }

    private void ApplyTurretPose()
    {
        Quaternion yawDelta = Quaternion.AngleAxis(currentYawDeg * yawDirection, yawLocalAxis.normalized);
        Quaternion pitchDelta = Quaternion.AngleAxis(currentPitchDeg * pitchDirection, pitchLocalAxis.normalized);

        turretYawPivot.localRotation = yawBaseRotation * yawDelta;
        hoodPitchPivot.localRotation = pitchBaseRotation * pitchDelta;
    }

    private void ShootVisibleBall()
    {
        Rigidbody ball = Instantiate(ballPrefab, muzzle.position, Quaternion.identity);
        activeBall = ball;

        ProjectileHitReporter reporter = ball.GetComponent<ProjectileHitReporter>();
        if (reporter != null)
        {
            reporter.Initialize(this);
        }

        float wheelRadiusMeters = 0.0508f;
        float exitSpeed = currentRpm * 2f * Mathf.PI / 60f * wheelRadiusMeters;

        Vector3 localForwardVelocity = muzzle.forward * exitSpeed;

        Vector3 inheritedVelocity = new Vector3(
            chassisVelocityXZ.x,
            0f,
            chassisVelocityXZ.y
        );

        ball.linearVelocity = localForwardVelocity + inheritedVelocity;
    }

    public void ReportProjectileHit()
    {
        hitReported = true;
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
        float rpm = 0f;

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
}
using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class SquareFieldBoundaryDrawer : MonoBehaviour
{
    [Header("FTC Field")]
    public float fieldSizeMeters = 3.6576f; // 12 ft in meters
    public float lineHeight = 0.02f;
    public float lineWidth = 0.04f;

    private LineRenderer lineRenderer;

    private void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
        DrawSquare();
    }

    private void OnValidate()
    {
        if (TryGetComponent(out LineRenderer lr))
        {
            lineRenderer = lr;
            DrawSquare();
        }
    }

    private void DrawSquare()
    {
        if (lineRenderer == null)
            return;

        float half = fieldSizeMeters * 0.5f;

        lineRenderer.positionCount = 5;
        lineRenderer.loop = false;
        lineRenderer.useWorldSpace = false;
        lineRenderer.widthMultiplier = lineWidth;

        lineRenderer.SetPosition(0, new Vector3(-half, lineHeight, -half));
        lineRenderer.SetPosition(1, new Vector3(-half, lineHeight, half));
        lineRenderer.SetPosition(2, new Vector3(half, lineHeight, half));
        lineRenderer.SetPosition(3, new Vector3(half, lineHeight, -half));
        lineRenderer.SetPosition(4, new Vector3(-half, lineHeight, -half));
    }
}
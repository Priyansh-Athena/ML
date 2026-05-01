using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class CircleBoundaryDrawer : MonoBehaviour
{
    public float radius = 3f;
    public int segments = 128;
    public float lineHeight = 0.02f;

    private LineRenderer lineRenderer;

    private void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
        DrawCircle();
    }

    private void OnValidate()
    {
        if (TryGetComponent(out LineRenderer lr))
        {
            lineRenderer = lr;
            DrawCircle();
        }
    }

    private void DrawCircle()
    {
        if (lineRenderer == null)
            return;

        lineRenderer.positionCount = segments + 1;
        lineRenderer.loop = true;
        lineRenderer.useWorldSpace = false;
        lineRenderer.widthMultiplier = 0.04f;

        for (int i = 0; i <= segments; i++)
        {
            float angle = i * Mathf.PI * 2f / segments;
            float x = Mathf.Cos(angle) * radius;
            float z = Mathf.Sin(angle) * radius;

            lineRenderer.SetPosition(i, new Vector3(x, lineHeight, z));
        }
    }
}
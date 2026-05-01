using UnityEngine;

public class ProjectileHitReporter : MonoBehaviour
{
    private TurretShooterAgent ownerAgent;

    public void Initialize(TurretShooterAgent agent)
    {
        ownerAgent = agent;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.transform.CompareTag("Target"))
        {
            ownerAgent.ReportProjectileHit();
            Destroy(gameObject);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            ownerAgent.ReportProjectileHit();
            Destroy(gameObject);
        }
    }
}
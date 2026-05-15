using UnityEngine;

public class ProjectileHitReporter : MonoBehaviour
{
    private SevenParameterMovingShooterAgent ownerAgent;

    public void Initialize(SevenParameterMovingShooterAgent agent)
    {
        ownerAgent = agent;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.transform.CompareTag("Target"))
        {
            if (ownerAgent != null)
                ownerAgent.ReportProjectileHit();

            Destroy(gameObject);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            if (ownerAgent != null)
                ownerAgent.ReportProjectileHit();

            Destroy(gameObject);
        }
    }
}
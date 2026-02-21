#include "actorconvexcallback.hpp"
#include "actor.hpp"
#include "collisiontype.hpp"

#include <BulletCollision/CollisionDispatch/btCollisionObject.h>
#include <components/misc/convert.hpp>

#include "projectile.hpp"

namespace MWPhysics
{
    namespace
    {
        // Geometric cylinder-cylinder overlap test. Replaces the expensive contactPairTest
        // which was the biggest per-actor cost in the solver.
        bool actorsOverlap(const btCollisionObject* a, const btCollisionObject* b)
        {
            auto* actorA = static_cast<const Actor*>(a->getUserPointer());
            auto* actorB = static_cast<const Actor*>(b->getUserPointer());
            const osg::Vec3f halfA = actorA->getHalfExtents();
            const osg::Vec3f halfB = actorB->getHalfExtents();
            const btVector3& originA = a->getWorldTransform().getOrigin();
            const btVector3& originB = b->getWorldTransform().getOrigin();
            const btVector3 delta = originA - originB;

            if (std::abs(delta.z()) >= halfA.z() + halfB.z())
                return false;

            const float dx = delta.x();
            const float dy = delta.y();
            const float combinedRadius = halfA.x() + halfB.x();
            return (dx * dx + dy * dy) < combinedRadius * combinedRadius;
        }
    }

    btScalar ActorConvexCallback::addSingleResult(
        btCollisionWorld::LocalConvexResult& convexResult, bool normalInWorldSpace)
    {
        if (convexResult.m_hitCollisionObject == mMe)
            return 1;

        // override data for actor-actor collisions
        // vanilla Morrowind seems to make overlapping actors collide as though they are both cylinders with a diameter
        // of the distance between them For some reason this doesn't work as well as it should when using capsules, but
        // it still helps a lot.
        if (convexResult.m_hitCollisionObject->getBroadphaseHandle()->m_collisionFilterGroup == CollisionType_Actor)
        {
            if (actorsOverlap(mMe, convexResult.m_hitCollisionObject))
            {
                auto originA = Misc::Convert::toOsg(mMe->getWorldTransform().getOrigin());
                auto originB = Misc::Convert::toOsg(convexResult.m_hitCollisionObject->getWorldTransform().getOrigin());
                osg::Vec3f motion = Misc::Convert::toOsg(mMotion);
                osg::Vec3f normal = (originA - originB);
                normal.z() = 0;
                normal.normalize();
                // only collide if horizontally moving towards the hit actor (note: the motion vector appears to be
                // inverted)
                // FIXME: This kinda screws with standing on actors that walk up slopes for some reason. Makes you fall
                // through them. It happens in vanilla Morrowind too, but much less often. I tried hunting down why but
                // couldn't figure it out. Possibly a stair stepping or ground ejection bug.
                if (normal * motion > 0.0f)
                {
                    convexResult.m_hitFraction = 0.0f;
                    convexResult.m_hitNormalLocal = Misc::Convert::toBullet(normal);
                    return ClosestConvexResultCallback::addSingleResult(convexResult, true);
                }
                else
                {
                    return 1;
                }
            }
        }
        if (convexResult.m_hitCollisionObject->getBroadphaseHandle()->m_collisionFilterGroup
            == CollisionType_Projectile)
        {
            auto* projectileHolder = static_cast<Projectile*>(convexResult.m_hitCollisionObject->getUserPointer());
            if (!projectileHolder->isActive())
                return 1;
            if (projectileHolder->isValidTarget(mMe))
                projectileHolder->hit(mMe, convexResult.m_hitPointLocal, convexResult.m_hitNormalLocal);
            return 1;
        }

        btVector3 hitNormalWorld;
        if (normalInWorldSpace)
            hitNormalWorld = convexResult.m_hitNormalLocal;
        else
        {
            /// need to transform normal into worldspace
            hitNormalWorld
                = convexResult.m_hitCollisionObject->getWorldTransform().getBasis() * convexResult.m_hitNormalLocal;
        }

        // dot product of the motion vector against the collision contact normal
        btScalar dotCollision = mMotion.dot(hitNormalWorld);
        if (dotCollision <= mMinCollisionDot)
            return 1;

        return ClosestConvexResultCallback::addSingleResult(convexResult, normalInWorldSpace);
    }
}

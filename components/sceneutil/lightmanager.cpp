#include "lightmanager.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>

#include <osg/BufferIndexBinding>
#include <osg/BufferObject>
#include <osg/Endian>
#include <osg/ValueObject>

#include <osgUtil/CullVisitor>

#include <components/misc/constants.hpp>
#include <components/misc/hash.hpp>
#include <components/resource/scenemanager.hpp>
#include <components/sceneutil/glextensions.hpp>
#include <components/sceneutil/util.hpp>
#include <components/settings/values.hpp>
#include <components/shader/shadermanager.hpp>

#include <components/debug/debuglog.hpp>
#include <osg/io_utils>

namespace
{
    constexpr int ffpMaxLights = 8;

    void configurePosition(osg::Matrixf& mat, const osg::Vec4& pos)
    {
        mat(0, 0) = pos.x();
        mat(0, 1) = pos.y();
        mat(0, 2) = pos.z();
    }

    void configureAmbient(osg::Matrixf& mat, const osg::Vec4& color)
    {
        mat(1, 0) = color.r();
        mat(1, 1) = color.g();
        mat(1, 2) = color.b();
    }

    void configureDiffuse(osg::Matrixf& mat, const osg::Vec4& color)
    {
        mat(2, 0) = color.r();
        mat(2, 1) = color.g();
        mat(2, 2) = color.b();
    }

    void configureSpecular(osg::Matrixf& mat, const osg::Vec4& color)
    {
        mat(3, 0) = color.r();
        mat(3, 1) = color.g();
        mat(3, 2) = color.b();
        mat(3, 3) = color.a();
    }

    void configureAttenuation(osg::Matrixf& mat, float c, float l, float q, float r)
    {
        mat(0, 3) = c;
        mat(1, 3) = l;
        mat(2, 3) = q;
        mat(3, 3) = r;
    }
}

namespace SceneUtil
{
    namespace
    {
        const std::unordered_map<std::string, LightingMethod> lightingMethodSettingMap = {
            { "legacy", LightingMethod::FFP },
            { "shaders compatibility", LightingMethod::PerObjectUniform },
            { "shaders", LightingMethod::SingleUBO },
        };
    }

    static int sLightId = 0;

    // Handles a GLSL shared layout by using configured offsets and strides to fill a continuous buffer, making the data
    // upload to GPU simpler.
    class LightBuffer : public osg::Referenced
    {
    public:
        enum LayoutOffset
        {
            Diffuse,
            DiffuseSign,
            Ambient,
            Specular,
            Position,
            AttenuationRadius
        };

        LightBuffer(int count)
            : mData(new osg::FloatArray(3 * 4 * count))
            , mEndian(osg::getCpuByteOrder())
            , mCount(count)
            , mCachedSunPos(osg::Vec4())
        {
        }

        LightBuffer(const LightBuffer&) = delete;

        void setDiffuse(int index, const osg::Vec4& value)
        {
            // Deal with negative lights (negative diffuse) by passing a sign bit in the unused alpha component
            auto positiveColor = value;
            unsigned int signBit = 1;
            if (value[0] < 0)
            {
                positiveColor *= -1.0;
                signBit = ~0u;
            }
            unsigned int packedColor = asRGBA(positiveColor);
            std::memcpy(&(*mData)[getOffset(index, Diffuse)], &packedColor, sizeof(unsigned int));
            std::memcpy(&(*mData)[getOffset(index, DiffuseSign)], &signBit, sizeof(unsigned int));
        }

        void setAmbient(int index, const osg::Vec4& value)
        {
            unsigned int packed = asRGBA(value);
            std::memcpy(&(*mData)[getOffset(index, Ambient)], &packed, sizeof(unsigned int));
        }

        void setSpecular(int index, const osg::Vec4& value)
        {
            unsigned int packed = asRGBA(value);
            std::memcpy(&(*mData)[getOffset(index, Specular)], &packed, sizeof(unsigned int));
        }

        void setPosition(int index, const osg::Vec4& value)
        {
            std::memcpy(&(*mData)[getOffset(index, Position)], value.ptr(), sizeof(osg::Vec4f));
        }

        void setAttenuationRadius(int index, const osg::Vec4& value)
        {
            std::memcpy(&(*mData)[getOffset(index, AttenuationRadius)], value.ptr(), sizeof(osg::Vec4f));
        }

        auto& getData() { return mData; }

        void dirty() { mData->dirty(); }

        static constexpr int queryBlockSize(int sz) { return 3 * osg::Vec4::num_components * sizeof(GLfloat) * sz; }

        void setCachedSunPos(const osg::Vec4& pos) { mCachedSunPos = pos; }

        void uploadCachedSunPos(const osg::Matrix& viewMat)
        {
            osg::Vec4 viewPos = mCachedSunPos * viewMat;
            std::memcpy(&(*mData)[getOffset(0, Position)], viewPos.ptr(), sizeof(osg::Vec4f));
        }

        unsigned int asRGBA(const osg::Vec4& value) const
        {
            return mEndian == osg::BigEndian ? value.asABGR() : value.asRGBA();
        }

        int getOffset(int index, LayoutOffset slot) const { return mOffsets.get(index, slot); }

        void configureLayout(int offsetColors, int offsetPosition, int offsetAttenuationRadius, int size, int stride)
        {
            configureLayout(Offsets(offsetColors, offsetPosition, offsetAttenuationRadius, stride), size);
        }

        void configureLayout(const LightBuffer* other)
        {
            mOffsets = other->mOffsets;
            int size = static_cast<int>(other->mData->size());

            configureLayout(mOffsets, size);
        }

    private:
        class Offsets
        {
        public:
            Offsets()
                : mStride(12)
            {
                mValues[Diffuse] = 0;
                mValues[Ambient] = 1;
                mValues[Specular] = 2;
                mValues[DiffuseSign] = 3;
                mValues[Position] = 4;
                mValues[AttenuationRadius] = 8;
            }

            Offsets(int offsetColors, int offsetPosition, int offsetAttenuationRadius, int stride)
                : mStride((offsetAttenuationRadius + sizeof(GLfloat) * osg::Vec4::num_components + stride) / 4)
            {
                constexpr auto sizeofFloat = sizeof(GLfloat);
                const auto diffuseOffset = static_cast<int>(offsetColors / sizeofFloat);

                mValues[Diffuse] = diffuseOffset;
                mValues[Ambient] = diffuseOffset + 1;
                mValues[Specular] = diffuseOffset + 2;
                mValues[DiffuseSign] = diffuseOffset + 3;
                mValues[Position] = offsetPosition / sizeofFloat;
                mValues[AttenuationRadius] = offsetAttenuationRadius / sizeofFloat;
            }

            int get(int index, LayoutOffset slot) const { return mStride * index + mValues[slot]; }

        private:
            int mStride;
            std::array<int, 6> mValues;
        };

        void configureLayout(const Offsets& offsets, int size)
        {
            // Copy cloned data using current layout into current data using new layout.
            // This allows to preserve osg::FloatArray buffer object in mData.
            const auto data = mData->asVector();
            mData->resizeArray(static_cast<unsigned>(size));
            for (int i = 0; i < mCount; ++i)
            {
                std::memcpy(
                    &(*mData)[offsets.get(i, Diffuse)], data.data() + getOffset(i, Diffuse), sizeof(osg::Vec4f));
                std::memcpy(
                    &(*mData)[offsets.get(i, Position)], data.data() + getOffset(i, Position), sizeof(osg::Vec4f));
                std::memcpy(&(*mData)[offsets.get(i, AttenuationRadius)], data.data() + getOffset(i, AttenuationRadius),
                    sizeof(osg::Vec4f));
            }
            mOffsets = offsets;
        }

        osg::ref_ptr<osg::FloatArray> mData;
        osg::Endian mEndian;
        int mCount;
        Offsets mOffsets;
        osg::Vec4 mCachedSunPos;
    };

    struct LightStateCache
    {
        std::vector<osg::Light*> lastAppliedLight;
    };

    LightStateCache* getLightStateCache(size_t contextid, size_t size = 8)
    {
        static std::vector<LightStateCache> cacheVector;
        if (cacheVector.size() < contextid + 1)
            cacheVector.resize(contextid + 1);
        cacheVector[contextid].lastAppliedLight.resize(size);
        return &cacheVector[contextid];
    }

    void configureStateSetSunOverride(
        LightManager* lightManager, const osg::Light* light, osg::StateSet* stateset, int mode)
    {
        auto method = lightManager->getLightingMethod();
        switch (method)
        {
            case LightingMethod::FFP:
            {
                break;
            }
            case LightingMethod::PerObjectUniform:
            {
                osg::Matrixf lightMat;
                configurePosition(lightMat, light->getPosition());
                configureAmbient(lightMat, light->getAmbient());
                configureDiffuse(lightMat, light->getDiffuse());
                configureSpecular(lightMat, light->getSpecular());

                stateset->addUniform(lightManager->generateLightBufferUniform(lightMat), mode);
                break;
            }
            case LightingMethod::SingleUBO:
            {
                osg::ref_ptr<LightBuffer> buffer = new LightBuffer(lightManager->getMaxLightsInScene());

                buffer->setDiffuse(0, light->getDiffuse());
                buffer->setAmbient(0, light->getAmbient());
                buffer->setSpecular(0, light->getSpecular());
                buffer->setPosition(0, light->getPosition());

                osg::ref_ptr<osg::UniformBufferObject> ubo = new osg::UniformBufferObject;
                buffer->getData()->setBufferObject(ubo);
                osg::ref_ptr<osg::UniformBufferBinding> ubb
                    = new osg::UniformBufferBinding(static_cast<int>(Resource::SceneManager::UBOBinding::LightBuffer),
                        buffer->getData(), 0, buffer->getData()->getTotalDataSize());
                stateset->setAttributeAndModes(ubb, mode);

                break;
            }
        }
    }

    class DisableLight : public osg::StateAttribute
    {
    public:
        DisableLight()
            : mIndex(0)
        {
        }
        DisableLight(int index)
            : mIndex(index)
        {
        }

        DisableLight(const DisableLight& copy, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY)
            : osg::StateAttribute(copy, copyop)
            , mIndex(copy.mIndex)
        {
        }

        META_StateAttribute(SceneUtil, DisableLight, osg::StateAttribute::LIGHT)

        unsigned int getMember() const override { return static_cast<unsigned int>(mIndex); }

        bool getModeUsage(ModeUsage& usage) const override
        {
            usage.usesMode(static_cast<GLMode>(GL_LIGHT0 + mIndex));
            return true;
        }

        int compare(const StateAttribute& sa) const override
        {
            throw std::runtime_error("DisableLight::compare: unimplemented");
        }

        void apply(osg::State& state) const override
        {
            GLenum lightNum = static_cast<GLenum>(GL_LIGHT0 + mIndex);
            glLightfv(lightNum, GL_AMBIENT, mNullptr.ptr());
            glLightfv(lightNum, GL_DIFFUSE, mNullptr.ptr());
            glLightfv(lightNum, GL_SPECULAR, mNullptr.ptr());

            LightStateCache* cache = getLightStateCache(state.getContextID());
            cache->lastAppliedLight[mIndex] = nullptr;
        }

    private:
        size_t mIndex;
        osg::Vec4f mNullptr;
    };

    class FFPLightStateAttribute : public osg::StateAttribute
    {
    public:
        FFPLightStateAttribute()
            : mIndex(0)
        {
        }
        FFPLightStateAttribute(size_t index, const std::vector<osg::ref_ptr<osg::Light>>& lights)
            : mIndex(index)
            , mLights(lights)
        {
        }

        FFPLightStateAttribute(
            const FFPLightStateAttribute& copy, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY)
            : osg::StateAttribute(copy, copyop)
            , mIndex(copy.mIndex)
            , mLights(copy.mLights)
        {
        }

        unsigned int getMember() const override { return static_cast<unsigned int>(mIndex); }

        bool getModeUsage(ModeUsage& usage) const override
        {
            for (size_t i = 0; i < mLights.size(); ++i)
                usage.usesMode(static_cast<GLMode>(GL_LIGHT0 + mIndex + i));
            return true;
        }

        int compare(const StateAttribute& sa) const override
        {
            throw std::runtime_error("FFPLightStateAttribute::compare: unimplemented");
        }

        META_StateAttribute(SceneUtil, FFPLightStateAttribute, osg::StateAttribute::LIGHT)

        void apply(osg::State& state) const override
        {
            if (mLights.empty())
                return;
            osg::Matrix modelViewMatrix = state.getModelViewMatrix();

            state.applyModelViewMatrix(state.getInitialViewMatrix());

            LightStateCache* cache = getLightStateCache(state.getContextID());

            for (size_t i = 0; i < mLights.size(); ++i)
            {
                osg::Light* current = cache->lastAppliedLight[i + mIndex];
                if (current != mLights[i].get())
                {
                    applyLight((GLenum)((int)GL_LIGHT0 + i + mIndex), mLights[i].get());
                    cache->lastAppliedLight[i + mIndex] = mLights[i].get();
                }
            }

            state.applyModelViewMatrix(modelViewMatrix);
        }

        void applyLight(GLenum lightNum, const osg::Light* light) const
        {
            glLightfv(lightNum, GL_AMBIENT, light->getAmbient().ptr());
            glLightfv(lightNum, GL_DIFFUSE, light->getDiffuse().ptr());
            glLightfv(lightNum, GL_SPECULAR, light->getSpecular().ptr());
            glLightfv(lightNum, GL_POSITION, light->getPosition().ptr());
            // TODO: enable this once spot lights are supported
            // need to transform SPOT_DIRECTION by the world matrix?
            // glLightfv(lightNum, GL_SPOT_DIRECTION, light->getDirection().ptr());
            // glLightf(lightNum, GL_SPOT_EXPONENT, light->getSpotExponent());
            // glLightf(lightNum, GL_SPOT_CUTOFF, light->getSpotCutoff());
            glLightf(lightNum, GL_CONSTANT_ATTENUATION, light->getConstantAttenuation());
            glLightf(lightNum, GL_LINEAR_ATTENUATION, light->getLinearAttenuation());
            glLightf(lightNum, GL_QUADRATIC_ATTENUATION, light->getQuadraticAttenuation());
        }

    private:
        size_t mIndex;
        std::vector<osg::ref_ptr<osg::Light>> mLights;
    };

    // Utility function to compute intersection with Z-plane
    osg::Vec3 computeZPlaneIntersection(const osg::Vec3& A, const osg::Vec3& B, float zDistance)
    {
        osg::Vec3 ab = B - A;
        float t = (zDistance - A.z()) / ab.z();
        return A + ab * t;
    }

    osg::Vec3f screenToView(
        const osg::Vec2f& screenCoord, const osg::Vec2f& screenDim, const osg::Matrixf& inverseProjection)
    {
        osg::Vec4f ndc = osg::Vec4f(
            screenCoord.x() / screenDim.x() * 2.0 - 1.0, screenCoord.y() / screenDim.y() * 2.0 - 1.0, -1.0, 1.0);

        osg::Vec4f viewCoord = ndc * inverseProjection;

        return {
            viewCoord.x() / viewCoord.w(),
            viewCoord.y() / viewCoord.w(),
            viewCoord.z() / viewCoord.w(),
        };
    }

    // Returns the intersection point of an infinite line and a
    // plane perpendicular to the Z-axis
    osg::Vec3f lineIntersectionWithZPlane(const osg::Vec3f& startPoint, const osg::Vec3f& endPoint, float zDistance)
    {
        osg::Vec3f direction = endPoint - startPoint;
        osg::Vec3f normal = osg::Vec3f(0.0, 0.0, -1.0); // plane normal

        // skip check if the line is parallel to the plane.
        float t = (zDistance - (normal * startPoint)) / (normal * direction);
        // the parametric form of the line equation
        return { startPoint.x() + t * direction.x(), startPoint.y() + t * direction.y(),
            startPoint.z() + t * direction.z() };
    }

    osg::Vec3f clamp(const osg::Vec3f x, const osg::Vec3f& min, const osg::Vec3f& max)
    {
        return osg::Vec3f(std::clamp(x.x(), min.x(), max.x()), std::clamp(x.y(), min.y(), max.y()),
            std::clamp(x.z(), min.z(), max.z()));
    }

    osg::Vec3f min(const osg::Vec3f& a, const osg::Vec3f& b)
    {
        return osg::Vec3f(std::min(a.x(), b.x()), std::min(a.y(), b.y()), std::min(a.z(), b.z()));
    }

    osg::Vec3f max(const osg::Vec3f& a, const osg::Vec3f& b)
    {
        return osg::Vec3f(std::max(a.x(), b.x()), std::max(a.y(), b.y()), std::max(a.z(), b.z()));
    }

    Cluster computeFroxelBounds(int x, int y, int z, int gridWidth, int gridHeight, int gridDepth, float zNear,
        float zFar, osgUtil::CullVisitor* cv)
    {
        osg::Vec2f screenDim = osg::Vec2f(cv->getViewport()->width(), cv->getViewport()->height());

        Cluster cluster;

        osg::Vec2 tileSize = osg::Vec2f(screenDim.x() / gridWidth, screenDim.y() / gridHeight);

        // Tile in screen-space
        osg::Vec2f minTile_screenspace = osg::Vec2f(x * tileSize.x(), y * tileSize.y());
        osg::Vec2f maxTile_screenspace = osg::Vec2f((x + 1) * tileSize.x(), (y + 1) * tileSize.y());

        osg::Matrixf inverseProjection = osg::Matrixf::inverse(*cv->getProjectionMatrix());

        // convert tile to view space sitting on the near plane
        osg::Vec3f minTile = screenToView(minTile_screenspace, screenDim, inverseProjection);
        osg::Vec3f maxTile = screenToView(maxTile_screenspace, screenDim, inverseProjection);

        float planeNear = zNear * std::pow(zFar / zNear, z / float(gridDepth));
        float planeFar = zNear * std::pow(zFar / zNear, (z + 1) / float(gridDepth));

        // the line goes from the eye position in view space (0, 0, 0)
        // through the min/max points of a tile to intersect with a given cluster's near-far planes
        osg::Vec3f minPointNear = lineIntersectionWithZPlane(osg::Vec3f(0, 0, 0), minTile, planeNear);
        osg::Vec3f minPointFar = lineIntersectionWithZPlane(osg::Vec3f(0, 0, 0), minTile, planeFar);
        osg::Vec3f maxPointNear = lineIntersectionWithZPlane(osg::Vec3f(0, 0, 0), maxTile, planeNear);
        osg::Vec3f maxPointFar = lineIntersectionWithZPlane(osg::Vec3f(0, 0, 0), maxTile, planeFar);

        cluster.minPoint = min(minPointNear, minPointFar);
        cluster.maxPoint = max(maxPointNear, maxPointFar);

        return cluster;
    }

    void LightManager::generateFroxelGrid(osgUtil::CullVisitor* cv)
    {
        osg::observer_ptr<osg::Camera> camPtr = cv->getCurrentCamera();

        // Check for any shared frustums we can use
        // For example, the refraction camera and main scene do not need seperate views/states
        for (const auto& [camera, viewData] : mCachedFroxelBounds)
        {
            if (camPtr.get() == camera.get())
                continue;

            // HACK: share refraction and scene camera view data/light assignments
            // Comparing the view/projection matrices doesn't work here
            if (camera->getName() == Constants::SceneCamera && camPtr->getName() == "RefractionCamera")
            {
                // Log(Debug::Error) << "FOUND: " << camera->getName() << " === " << camPtr->getName();

                mCachedFroxelBounds[camPtr] = viewData;
                return;
            }
        }

        const osg::RefMatrix* viewMatrix = cv->getCurrentRenderStage()->getInitialViewMatrix();

        int gridWidth = 16;
        int gridHeight = 9;
        int gridDepth = 24;
        float zNear = Settings::camera().mNearClip;
        float zFar = Settings::camera().mViewingDistance;

        const auto& lights = getLightsInViewSpace(cv, viewMatrix, cv->getTraversalNumber());

        auto& indexMap = getLightIndexMap(cv->getTraversalNumber());
        for (const auto& light : lights)
        {
            auto id = light.mLightSource->getId();
            if (indexMap.find(id) != indexMap.end())
                continue;

            int index = indexMap.size() + 1;
            updateGPUPointLight(index, light.mLightSource, cv->getTraversalNumber(), viewMatrix);
            indexMap.emplace(id, index);
        }

        auto it = mCachedFroxelBounds.find(camPtr);
        // Generate froxel bounds for this camera if we haven't already
        // TODO: need to re-generate if the camera properties change (width, height, fov, near, far)
        if (it == mCachedFroxelBounds.end())
        {
            std::vector<Cluster> bounds(gridWidth * gridHeight * gridDepth);

            for (int x = 0; x < gridWidth; ++x)
            {
                for (int y = 0; y < gridHeight; ++y)
                {
                    for (int z = 0; z < gridDepth; ++z)
                    {
                        int index = x + y * gridWidth + z * (gridWidth * gridHeight);
                        bounds[index] = computeFroxelBounds(
                            x, y, z, gridWidth, gridHeight, gridDepth, Settings::camera().mNearClip, 8192, cv);
                    }
                }
            }

            it = mCachedFroxelBounds.insert(std::make_pair(camPtr, std::make_shared<ViewDependentFroxel>())).first;
            it->second->mFroxelData = std::move(bounds);
        }

        auto& froxelGrid = it->second->mAttribute[cv->getTraversalNumber() % 2]->mFroxelGrid;
        auto& lightIndices = it->second->mAttribute[cv->getTraversalNumber() % 2]->mLightIndices;

        froxelGrid.clear();
        lightIndices.clear();

        // zero is sun
        lightIndices.push_back(0);

        int froxelCount = gridWidth * gridHeight * gridDepth;
        froxelGrid.resize(froxelCount);

        const auto& froxels = it->second->mFroxelData;

        const int numThreads = std::thread::hardware_concurrency();

        // Thread-safe storage for light indices
        std::vector<std::vector<std::uint32_t>> froxelLightIndices(gridWidth * gridHeight * gridDepth);

        // Divide work into chunks
        auto processChunk = [&](int startFroxel, int endFroxel) {
            for (int froxelID = startFroxel; froxelID < endFroxel; ++froxelID)
            {
                const Cluster& cluster = froxels[froxelID];

                std::vector<std::uint32_t> localLightIndices;

                for (size_t i = 0; i < lights.size(); ++i)
                {
                    osg::Vec3f center = lights[i].mViewBound.center();
                    float radius = lights[i].mLightSource->getRadius() * 2;

                    osg::Vec3f closestPoint = clamp(center, cluster.minPoint, cluster.maxPoint);
                    float distanceSquared = (closestPoint - center).length2();

                    if (distanceSquared < radius * radius)
                    {
                        localLightIndices.push_back(static_cast<std::uint32_t>(i + 1));
                    }
                }

                froxelLightIndices[froxelID] = std::move(localLightIndices);
            }
        };

        std::vector<std::thread> threads;
        int totalFroxels = gridWidth * gridHeight * gridDepth;
        int chunkSize = (totalFroxels + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; ++t)
        {
            int startFroxel = t * chunkSize;
            int endFroxel = std::min(startFroxel + chunkSize, totalFroxels);
            threads.emplace_back(processChunk, startFroxel, endFroxel);
        }

        for (auto& thread : threads)
            thread.join();

        for (int froxelID = 0; froxelID < froxelLightIndices.size(); ++froxelID)
        {
            froxelGrid[froxelID].offset = lightIndices.size();
            lightIndices.insert(
                lightIndices.end(), froxelLightIndices[froxelID].begin(), froxelLightIndices[froxelID].end());
            froxelGrid[froxelID].count = froxelLightIndices[froxelID].size();
        }
    }

    struct StateSetGenerator
    {
        LightManager* mLightManager;

        virtual ~StateSetGenerator() = default;

        virtual osg::ref_ptr<osg::StateSet> generate(const LightManager::LightList& lightList, size_t frameNum) = 0;

        virtual void update(osg::StateSet* stateset, const LightManager::LightList& lightList, size_t frameNum) {}

        osg::Matrix mViewMatrix;
    };

    struct StateSetGeneratorFFP : StateSetGenerator
    {
        osg::ref_ptr<osg::StateSet> generate(const LightManager::LightList& lightList, size_t frameNum) override
        {
            osg::ref_ptr<osg::StateSet> stateset = new osg::StateSet;

            std::vector<osg::ref_ptr<osg::Light>> lights;
            lights.reserve(lightList.size());
            for (size_t i = 0; i < lightList.size(); ++i)
                lights.emplace_back(lightList[i]->mLightSource->getLight(frameNum));

            // the first light state attribute handles the actual state setting for all lights
            // it's best to batch these up so that we don't need to touch the modelView matrix more than necessary
            // don't use setAttributeAndModes, that does not support light indices!
            stateset->setAttribute(
                new FFPLightStateAttribute(mLightManager->getStartLight(), std::move(lights)), osg::StateAttribute::ON);

            for (size_t i = 0; i < lightList.size(); ++i)
                stateset->setMode(
                    GL_LIGHT0 + mLightManager->getStartLight() + static_cast<int>(i), osg::StateAttribute::ON);

            // need to push some dummy attributes to ensure proper state tracking
            // lights need to reset to their default when the StateSet is popped
            for (size_t i = 1; i < lightList.size(); ++i)
                stateset->setAttribute(
                    mLightManager->getDummies()[i + mLightManager->getStartLight()].get(), osg::StateAttribute::ON);

            return stateset;
        }
    };

    struct StateSetGeneratorPerObjectUniform : StateSetGenerator
    {
        osg::ref_ptr<osg::StateSet> generate(const LightManager::LightList& lightList, size_t frameNum) override
        {
            osg::ref_ptr<osg::StateSet> stateset = new osg::StateSet;
            osg::ref_ptr<osg::Uniform> data
                = mLightManager->generateLightBufferUniform(mLightManager->getSunlightBuffer(frameNum));

            for (size_t i = 0; i < lightList.size(); ++i)
            {
                auto* light = lightList[i]->mLightSource->getLight(frameNum);
                osg::Matrixf lightMat;
                configurePosition(lightMat, light->getPosition() * mViewMatrix);
                configureAmbient(lightMat, light->getAmbient());
                configureDiffuse(lightMat, light->getDiffuse());
                configureSpecular(lightMat, light->getSpecular());
                configureAttenuation(lightMat, light->getConstantAttenuation(), light->getLinearAttenuation(),
                    light->getQuadraticAttenuation(), lightList[i]->mLightSource->getRadius());

                data->setElement(static_cast<unsigned int>(i + 1), lightMat);
            }

            stateset->addUniform(data);
            stateset->addUniform(new osg::Uniform("PointLightCount", static_cast<int>(lightList.size() + 1)));

            return stateset;
        }
    };

    LightManager* findLightManager(const osg::NodePath& path)
    {
        for (size_t i = 0; i < path.size(); ++i)
        {
            if (LightManager* lightManager = dynamic_cast<LightManager*>(path[i]))
                return lightManager;
        }
        return nullptr;
    }

    // Set on a LightSource. Adds the light source to its light manager for the current frame.
    // This allows us to keep track of the current lights in the scene graph without tying creation & destruction to the
    // manager.
    class CollectLightCallback : public NodeCallback<CollectLightCallback>
    {
    public:
        CollectLightCallback()
            : mLightManager(nullptr)
        {
        }

        CollectLightCallback(const CollectLightCallback& copy, const osg::CopyOp& copyop)
            : NodeCallback<CollectLightCallback>(copy, copyop)
            , mLightManager(nullptr)
        {
        }

        META_Object(SceneUtil, CollectLightCallback)

        void operator()(osg::Node* node, osg::NodeVisitor* nv)
        {
            if (!mLightManager)
            {
                mLightManager = findLightManager(nv->getNodePath());

                if (!mLightManager)
                    throw std::runtime_error("can't find parent LightManager");
            }

            mLightManager->addLight(
                static_cast<LightSource*>(node), osg::computeLocalToWorld(nv->getNodePath()), nv->getTraversalNumber());

            traverse(node, nv);
        }

    private:
        LightManager* mLightManager;
    };

    // Set on a LightManager. Clears the data from the previous frame.
    class LightManagerUpdateCallback : public SceneUtil::NodeCallback<LightManagerUpdateCallback>
    {
    public:
        void operator()(osg::Node* node, osg::NodeVisitor* nv)
        {
            LightManager* lightManager = static_cast<LightManager*>(node);
            lightManager->update(nv->getTraversalNumber());

            traverse(node, nv);
        }
    };

    class LightManagerCullCallback
        : public SceneUtil::NodeCallback<LightManagerCullCallback, LightManager*, osgUtil::CullVisitor*>
    {
    public:
        LightManagerCullCallback(LightManager* lightManager)
        {
            if (!lightManager->getUBOManager())
                return;

            for (size_t i = 0; i < mUBBs.size(); ++i)
            {
                auto& buffer = lightManager->getUBOManager()->getLightBuffer(i);
                mUBBs[i]
                    = new osg::UniformBufferBinding(static_cast<int>(Resource::SceneManager::UBOBinding::LightBuffer),
                        buffer->getData(), 0, buffer->getData()->getTotalDataSize());
            }
        }

        void operator()(LightManager* node, osgUtil::CullVisitor* cv)
        {
            if (!(cv->getTraversalMask() & node->getLightingMask()))
            {
                traverse(node, cv);
                return;
            }

            osg::ref_ptr<osg::StateSet> stateset = new osg::StateSet;

            if (node->getLightingMethod() == LightingMethod::SingleUBO)
            {
                node->generateFroxelGrid(cv);

                stateset->setAttributeAndModes(
                    node->mCachedFroxelBounds[cv->getCurrentCamera()]->mAttribute[cv->getTraversalNumber() % 2],
                    osg::StateAttribute::ON);
                const size_t frameId = cv->getTraversalNumber() % 2;
                stateset->setAttributeAndModes(mUBBs[frameId], osg::StateAttribute::ON);

                auto& buffer = node->getUBOManager()->getLightBuffer(cv->getTraversalNumber());

                if (auto sun = node->getSunlight())
                {
                    buffer->setCachedSunPos(sun->getPosition());
                    buffer->setAmbient(0, sun->getAmbient());
                    buffer->setDiffuse(0, sun->getDiffuse());
                    buffer->setSpecular(0, sun->getSpecular());
                }
            }
            else if (node->getLightingMethod() == LightingMethod::PerObjectUniform)
            {
                if (auto sun = node->getSunlight())
                {
                    osg::Matrixf lightMat;
                    configurePosition(
                        lightMat, sun->getPosition() * (*cv->getCurrentRenderStage()->getInitialViewMatrix()));
                    configureAmbient(lightMat, sun->getAmbient());
                    configureDiffuse(lightMat, sun->getDiffuse());
                    configureSpecular(lightMat, sun->getSpecular());
                    node->setSunlightBuffer(lightMat, cv->getTraversalNumber());
                    stateset->addUniform(node->generateLightBufferUniform(lightMat));
                }
            }
            cv->pushStateSet(stateset);
            traverse(node, cv);
            cv->popStateSet();

            if (node->getPPLightsBuffer() && cv->getCurrentCamera()->getName() == Constants::SceneCamera)
                node->getPPLightsBuffer()->updateCount(cv->getTraversalNumber());
        }

        std::array<osg::ref_ptr<osg::UniformBufferBinding>, 2> mUBBs;
    };

    UBOManager::UBOManager(int lightCount)
        : mDummyProgram(new osg::Program)
        , mInitLayout(false)
        , mDirty({ true, true })
        , mTemplate(new LightBuffer(lightCount))
    {
        static const std::string dummyVertSource = generateDummyShader(lightCount);

        // Needed to query the layout of the buffer object. The layout specifier needed to use the std140 layout is not
        // reliably available, regardless of extensions, until GLSL 140.
        mDummyProgram->addShader(new osg::Shader(osg::Shader::VERTEX, dummyVertSource));
        mDummyProgram->addBindUniformBlock(
            "LightBufferBinding", static_cast<int>(Resource::SceneManager::UBOBinding::LightBuffer));

        for (size_t i = 0; i < mLightBuffers.size(); ++i)
        {
            mLightBuffers[i] = new LightBuffer(lightCount);

            osg::ref_ptr<osg::UniformBufferObject> ubo = new osg::UniformBufferObject;
            ubo->setUsage(GL_STREAM_DRAW);

            mLightBuffers[i]->getData()->setBufferObject(ubo);
        }
    }

    UBOManager::UBOManager(const UBOManager& copy, const osg::CopyOp& copyop)
        : osg::StateAttribute(copy, copyop)
        , mDummyProgram(copy.mDummyProgram)
        , mInitLayout(copy.mInitLayout)
    {
    }

    void UBOManager::releaseGLObjects(osg::State* state) const
    {
        mDummyProgram->releaseGLObjects(state);
    }

    int UBOManager::compare(const StateAttribute& sa) const
    {
        throw std::runtime_error("LightManagerStateAttribute::compare: unimplemented");
    }

    void UBOManager::apply(osg::State& state) const
    {
        unsigned int frame = state.getFrameStamp()->getFrameNumber();
        unsigned int index = frame % 2;

        auto* ext = state.get<osg::GLExtensions>();

        if (!mInitLayout)
        {
            mDummyProgram->apply(state);
            auto handle = mDummyProgram->getPCP(state)->getHandle();

            int activeUniformBlocks = 0;
            ext->glGetProgramiv(handle, GL_ACTIVE_UNIFORM_BLOCKS, &activeUniformBlocks);

            // wait until the UBO binding is created
            if (activeUniformBlocks > 0)
            {
                initSharedLayout(ext, handle, frame);
                mInitLayout = true;
            }
        }
        else if (mDirty[index])
        {
            mDirty[index] = false;
            mLightBuffers[index]->configureLayout(mTemplate);
        }

        mLightBuffers[index]->uploadCachedSunPos(state.getInitialViewMatrix());
        mLightBuffers[index]->dirty();
    }

    std::string UBOManager::generateDummyShader(int maxLightsInScene)
    {
        const std::string define = "@maxLightsInScene";

        std::string shader = R"GLSL(
            #version 120
            #extension GL_ARB_uniform_buffer_object : require
            struct LightData {
                ivec4 packedColors;
                vec4 position;
                vec4 attenuation;
            };
            uniform LightBufferBinding {
                LightData LightBuffer[@maxLightsInScene];
            };
            void main()
            {
                gl_Position = vec4(0.0);
            }
        )GLSL";

        shader.replace(shader.find(define), define.length(), std::to_string(maxLightsInScene));
        return shader;
    }

    void UBOManager::initSharedLayout(osg::GLExtensions* ext, int handle, unsigned int frame) const
    {
        constexpr std::array<unsigned int, 1> index
            = { static_cast<unsigned int>(Resource::SceneManager::UBOBinding::LightBuffer) };
        int totalBlockSize = -1;
        int stride = -1;

        ext->glGetActiveUniformBlockiv(handle, 0, GL_UNIFORM_BLOCK_DATA_SIZE, &totalBlockSize);
        ext->glGetActiveUniformsiv(
            handle, static_cast<GLsizei>(index.size()), index.data(), GL_UNIFORM_ARRAY_STRIDE, &stride);

        std::array<const char*, 3> names = {
            "LightBuffer[0].packedColors",
            "LightBuffer[0].position",
            "LightBuffer[0].attenuation",
        };
        std::vector<unsigned int> indices(names.size());
        std::vector<int> offsets(names.size());

        ext->glGetUniformIndices(handle, static_cast<GLsizei>(names.size()), names.data(), indices.data());
        ext->glGetActiveUniformsiv(
            handle, static_cast<GLsizei>(indices.size()), indices.data(), GL_UNIFORM_OFFSET, offsets.data());

        mTemplate->configureLayout(offsets[0], offsets[1], offsets[2], totalBlockSize, stride);
    }

    ClusteredLightsAttribute::ClusteredLightsAttribute(const ClusteredLightsAttribute& copy, const osg::CopyOp& copyop)
        : osg::StateAttribute(copy, copyop)
        , mFroxelGrid(copy.mFroxelGrid)
        , mLightIndices(copy.mLightIndices)
    {
    }

    void ClusteredLightsAttribute::apply(osg::State& state) const
    {
        unsigned int frame = state.getFrameStamp()->getFrameNumber();
        unsigned int index = frame % 2;

        auto* ext = state.get<osg::GLExtensions>();

        if (mFroxelBuffer == 0)
        {
            ext->glGenBuffers(1, &mFroxelBuffer);
            ext->glGenBuffers(1, &mLightIndexBuffer);
        }

        // Upload froxel grid
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mFroxelBuffer);

        if (mOldFroxelBufferSize != mFroxelGrid.size())
        {
            mOldFroxelBufferSize = mFroxelGrid.size();

            ext->glBufferData(
                GL_SHADER_STORAGE_BUFFER, mFroxelGrid.size() * sizeof(Froxel), mFroxelGrid.data(), GL_STATIC_DRAW);
        }
        else
        {
            ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mFroxelGrid.size() * sizeof(Froxel), mFroxelGrid.data());
        }

        // Upload light indices
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mLightIndexBuffer);

        if (mOldLightIndexBufferSize != mLightIndices.size())
        {
            mOldLightIndexBufferSize = mLightIndices.size();

            ext->glBufferData(GL_SHADER_STORAGE_BUFFER, mLightIndices.size() * sizeof(std::uint32_t),
                mLightIndices.data(), GL_DYNAMIC_DRAW);
        }
        else
        {
            ext->glBufferSubData(
                GL_SHADER_STORAGE_BUFFER, 0, mLightIndices.size() * sizeof(std::uint32_t), mLightIndices.data());
        }

        // Bind buffers to shader storage binding points
        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, mFroxelBuffer);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, mLightIndexBuffer);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    LightingMethod LightManager::getLightingMethodFromString(const std::string& value)
    {
        auto it = lightingMethodSettingMap.find(value);
        if (it != lightingMethodSettingMap.end())
            return it->second;

        constexpr const char* fallback = "shaders compatibility";
        Log(Debug::Warning) << "Unknown lighting method '" << value << "', returning fallback '" << fallback << "'";
        return LightingMethod::PerObjectUniform;
    }

    std::string LightManager::getLightingMethodString(LightingMethod method)
    {
        for (const auto& p : lightingMethodSettingMap)
            if (p.second == method)
                return p.first;
        return "";
    }

    LightManager::LightManager(const LightSettings& settings)
        : mStartLight(0)
        , mLightingMask(~0u)
        , mSun(nullptr)
        , mPointLightRadiusMultiplier(1.f)
        , mPointLightFadeEnd(0.f)
        , mPointLightFadeStart(0.f)
    {
        osg::GLExtensions* exts = SceneUtil::glExtensionsReady() ? &SceneUtil::getGLExtensions() : nullptr;
        bool supportsUBO = exts && exts->isUniformBufferObjectSupported;
        bool supportsGPU4 = exts && exts->isGpuShader4Supported;

        mSupported[static_cast<int>(LightingMethod::FFP)] = true;
        mSupported[static_cast<int>(LightingMethod::PerObjectUniform)] = true;
        mSupported[static_cast<int>(LightingMethod::SingleUBO)] = supportsUBO && supportsGPU4;

        setUpdateCallback(new LightManagerUpdateCallback);

        if (settings.mLightingMethod == LightingMethod::FFP)
        {
            initFFP(ffpMaxLights);
        }
        else
        {
            static bool hasLoggedWarnings = false;

            if (settings.mLightingMethod == LightingMethod::SingleUBO && !hasLoggedWarnings)
            {
                if (!supportsUBO)
                    Log(Debug::Warning) << "GL_ARB_uniform_buffer_object not supported: switching to shader "
                                           "compatibility lighting mode";
                if (!supportsGPU4)
                    Log(Debug::Warning)
                        << "GL_EXT_gpu_shader4 not supported: switching to shader compatibility lighting mode";
                hasLoggedWarnings = true;
            }

            if (!supportsUBO || !supportsGPU4 || settings.mLightingMethod == LightingMethod::PerObjectUniform)
                initPerObjectUniform(settings.mMaxLights);
            else
                initSingleUBO(settings.mMaxLights);

            getOrCreateStateSet()->addUniform(new osg::Uniform("PointLightCount", 0));

            addCullCallback(new LightManagerCullCallback(this));
        }

        updateSettings(settings.mLightBoundsMultiplier, settings.mMaximumLightDistance, settings.mLightFadeStart);
    }

    LightManager::LightManager(const LightManager& copy, const osg::CopyOp& copyop)
        : osg::Group(copy, copyop)
        , mStartLight(copy.mStartLight)
        , mLightingMask(copy.mLightingMask)
        , mSun(copy.mSun)
        , mLightingMethod(copy.mLightingMethod)
        , mPointLightRadiusMultiplier(copy.mPointLightRadiusMultiplier)
        , mPointLightFadeEnd(copy.mPointLightFadeEnd)
        , mPointLightFadeStart(copy.mPointLightFadeStart)
        , mMaxLights(copy.mMaxLights)
        , mPPLightBuffer(copy.mPPLightBuffer)
    {
    }

    LightingMethod LightManager::getLightingMethod() const
    {
        return mLightingMethod;
    }

    bool LightManager::usingFFP() const
    {
        return mLightingMethod == LightingMethod::FFP;
    }

    int LightManager::getMaxLights() const
    {
        return mMaxLights;
    }

    void LightManager::setMaxLights(int value)
    {
        mMaxLights = value;
    }

    int LightManager::getMaxLightsInScene() const
    {
        static constexpr int max = 16384 / LightBuffer::queryBlockSize(1);
        return max;
    }

    Shader::ShaderManager::DefineMap LightManager::getLightDefines() const
    {
        Shader::ShaderManager::DefineMap defines;

        defines["maxLights"] = std::to_string(getMaxLights());
        defines["maxLightsInScene"] = std::to_string(getMaxLightsInScene());
        defines["lightingMethodFFP"] = getLightingMethod() == LightingMethod::FFP ? "1" : "0";
        defines["lightingMethodPerObjectUniform"] = getLightingMethod() == LightingMethod::PerObjectUniform ? "1" : "0";
        defines["lightingMethodUBO"] = getLightingMethod() == LightingMethod::SingleUBO ? "1" : "0";
        defines["useUBO"] = std::to_string(getLightingMethod() == LightingMethod::SingleUBO);
        // exposes bitwise operators
        defines["useGPUShader4"] = std::to_string(getLightingMethod() == LightingMethod::SingleUBO);
        defines["getLight"] = getLightingMethod() == LightingMethod::FFP ? "gl_LightSource" : "LightBuffer";
        defines["startLight"] = getLightingMethod() == LightingMethod::SingleUBO ? "0" : "1";
        defines["endLight"] = getLightingMethod() == LightingMethod::FFP ? defines["maxLights"] : "PointLightCount";

        return defines;
    }

    void LightManager::processChangedSettings(
        float lightBoundsMultiplier, float maximumLightDistance, float lightFadeStart)
    {
        updateSettings(lightBoundsMultiplier, maximumLightDistance, lightFadeStart);
    }

    void LightManager::updateMaxLights(int maxLights)
    {
        if (usingFFP())
            return;

        setMaxLights(maxLights);

        if (getLightingMethod() == LightingMethod::PerObjectUniform)
        {
            getStateSet()->removeUniform("LightBuffer");
            getStateSet()->addUniform(generateLightBufferUniform(osg::Matrixf()));
        }

        for (auto& cache : mStateSetCache)
            cache.clear();
    }

    void LightManager::updateSettings(float lightBoundsMultiplier, float maximumLightDistance, float lightFadeStart)
    {
        mPointLightRadiusMultiplier = lightBoundsMultiplier;
        mPointLightFadeEnd = maximumLightDistance;
        if (mPointLightFadeEnd > 0)
            mPointLightFadeStart = mPointLightFadeEnd * lightFadeStart;
    }

    void LightManager::initFFP(int targetLights)
    {
        setLightingMethod(LightingMethod::FFP);
        setMaxLights(targetLights);

        for (int i = 0; i < getMaxLights(); ++i)
            mDummies.push_back(new FFPLightStateAttribute(i, std::vector<osg::ref_ptr<osg::Light>>()));
    }

    void LightManager::initPerObjectUniform(int targetLights)
    {
        setLightingMethod(LightingMethod::PerObjectUniform);
        setMaxLights(targetLights);

        getOrCreateStateSet()->addUniform(generateLightBufferUniform(osg::Matrixf()));
    }

    void LightManager::initSingleUBO(int targetLights)
    {
        setLightingMethod(LightingMethod::SingleUBO);
        setMaxLights(targetLights);

        mUBOManager = new UBOManager(getMaxLightsInScene());
        getOrCreateStateSet()->setAttributeAndModes(mUBOManager);
    }

    void LightManager::setLightingMethod(LightingMethod method)
    {
        mLightingMethod = method;
        switch (method)
        {
            case LightingMethod::FFP:
                mStateSetGenerator = std::make_unique<StateSetGeneratorFFP>();
                break;
            case LightingMethod::PerObjectUniform:
                mStateSetGenerator = std::make_unique<StateSetGeneratorPerObjectUniform>();
                break;
        }

        if (mStateSetGenerator)
            mStateSetGenerator->mLightManager = this;
    }

    void LightManager::setLightingMask(size_t mask)
    {
        mLightingMask = mask;
    }

    size_t LightManager::getLightingMask() const
    {
        return mLightingMask;
    }

    void LightManager::setStartLight(int start)
    {
        mStartLight = start;

        if (!usingFFP())
            return;

        // Set default light state to zero
        // This is necessary because shaders don't respect glDisable(GL_LIGHTX) so in addition to disabling
        // we'll have to set a light state that has no visible effect
        for (int i = start; i < getMaxLights(); ++i)
        {
            osg::ref_ptr<DisableLight> defaultLight(new DisableLight(i));
            getOrCreateStateSet()->setAttributeAndModes(defaultLight, osg::StateAttribute::OFF);
        }
    }

    int LightManager::getStartLight() const
    {
        return mStartLight;
    }

    void LightManager::update(size_t frameNum)
    {
        if (mPPLightBuffer)
            mPPLightBuffer->clear(frameNum);

        getLightIndexMap(frameNum).clear();
        mLights.clear();
        mLightsInViewSpace.clear();

        // Do an occasional cleanup for orphaned lights.
        for (int i = 0; i < 2; ++i)
        {
            if (mStateSetCache[i].size() > 5000)
                mStateSetCache[i].clear();
        }
    }

    void LightManager::addLight(LightSource* lightSource, const osg::Matrixf& worldMat, size_t frameNum)
    {
        LightSourceTransform l;
        l.mLightSource = lightSource;
        l.mWorldMatrix = worldMat;
        osg::Vec3f pos = worldMat.getTrans();
        lightSource->getLight(frameNum)->setPosition(osg::Vec4f(pos, 1.f));

        mLights.push_back(l);
    }

    void LightManager::setSunlight(osg::ref_ptr<osg::Light> sun)
    {
        if (usingFFP())
            return;

        mSun = sun;
    }

    osg::ref_ptr<osg::Light> LightManager::getSunlight()
    {
        return mSun;
    }

    size_t LightManager::HashLightIdList::operator()(const LightIdList& lightIdList) const
    {
        size_t hash = 0;
        for (size_t i = 0; i < lightIdList.size(); ++i)
            Misc::hashCombine(hash, lightIdList[i]);
        return hash;
    }

    osg::ref_ptr<osg::StateSet> LightManager::getLightListStateSet(
        const LightList& lightList, size_t frameNum, const osg::RefMatrix* viewMatrix)
    {
        if (getLightingMethod() == LightingMethod::PerObjectUniform)
        {
            mStateSetGenerator->mViewMatrix = *viewMatrix;
            return mStateSetGenerator->generate(lightList, frameNum);
        }

        // possible optimization: return a StateSet containing all requested lights plus some extra lights (if a
        // suitable one exists)
        auto& stateSetCache = mStateSetCache[frameNum % 2];

        LightIdList lightIdList;
        lightIdList.reserve(lightList.size());
        std::transform(lightList.begin(), lightList.end(), std::back_inserter(lightIdList),
            [](const LightSourceViewBound* l) { return l->mLightSource->getId(); });

        auto found = stateSetCache.find(lightIdList);
        if (found != stateSetCache.end())
        {
            mStateSetGenerator->update(found->second, lightList, frameNum);
            return found->second;
        }

        auto stateset = mStateSetGenerator->generate(lightList, frameNum);
        stateSetCache.emplace(lightIdList, stateset);
        return stateset;
    }

    const std::vector<LightManager::LightSourceViewBound>& LightManager::getLightsInViewSpace(
        osgUtil::CullVisitor* cv, const osg::RefMatrix* viewMatrix, size_t frameNum)
    {
        osg::Camera* camera = cv->getCurrentCamera();

        osg::observer_ptr<osg::Camera> camPtr(camera);
        auto it = mLightsInViewSpace.find(camPtr);

        if (it == mLightsInViewSpace.end())
        {
            it = mLightsInViewSpace.insert(std::make_pair(camPtr, LightSourceViewBoundCollection())).first;

            for (const auto& transform : mLights)
            {
                osg::Matrixf worldViewMat = transform.mWorldMatrix * (*viewMatrix);

                float radius = transform.mLightSource->getRadius();

                osg::BoundingSphere viewBound(osg::Vec3f(), radius * mPointLightRadiusMultiplier);
                transformBoundingSphere(worldViewMat, viewBound);

                if (transform.mLightSource->getLastAppliedFrame() != frameNum && mPointLightFadeEnd != 0.f)
                {
                    const float fadeDelta = mPointLightFadeEnd - mPointLightFadeStart;
                    const float viewDelta = viewBound.center().length() - mPointLightFadeStart;
                    float fade = 1 - std::clamp(viewDelta / fadeDelta, 0.f, 1.f);
                    if (fade == 0.f)
                        continue;

                    auto* light = transform.mLightSource->getLight(frameNum);
                    light->setDiffuse(light->getDiffuse() * fade);
                    light->setSpecular(light->getSpecular() * fade);
                    transform.mLightSource->setLastAppliedFrame(frameNum);
                }

                osg::CullingSet& cullingSet = cv->getModelViewCullingStack().front();
                osg::BoundingSphere frustumBound = viewBound;
                frustumBound.radius() = transform.mLightSource->getRadius() * 2.f;
                // Ignore culled lights
                if (cullingSet.isCulled(frustumBound))
                    continue;

                LightSourceViewBound l;
                l.mLightSource = transform.mLightSource;
                l.mViewBound = viewBound;
                it->second.push_back(l);
            }

            const bool fillPPLights = mPPLightBuffer && it->first->getName() == Constants::SceneCamera;
            const bool sceneLimitReached = getLightingMethod() == LightingMethod::SingleUBO
                && it->second.size() > static_cast<size_t>(getMaxLightsInScene() - 1);

            if (fillPPLights || sceneLimitReached)
            {
                auto sorter = [](const LightSourceViewBound& left, const LightSourceViewBound& right) {
                    return left.mViewBound.center().length2() - left.mViewBound.radius2()
                        < right.mViewBound.center().length2() - right.mViewBound.radius2();
                };

                std::sort(it->second.begin(), it->second.end(), sorter);

                if (fillPPLights)
                {
                    // osg::CullingSet& cullingSet = cv->getModelViewCullingStack().front();
                    for (const auto& bound : it->second)
                    {
                        if (bound.mLightSource->getEmpty())
                            continue;
                        const auto* light = bound.mLightSource->getLight(frameNum);
                        // Ignore negative lights
                        if (light->getDiffuse().x() < 0.f)
                            continue;
                        const float radius = bound.mLightSource->getRadius();
                        // osg::BoundingSphere frustumBound = bound.mViewBound;
                        // frustumBound.radius() = radius * 2.f;
                        // // Ignore culled lights
                        // if (cullingSet.isCulled(frustumBound))
                        //     continue;
                        mPPLightBuffer->setLight(frameNum, light, radius);
                    }
                }

                if (sceneLimitReached)
                    it->second.resize(getMaxLightsInScene() - 1);
            }
        }

        return it->second;
    }

    void LightManager::updateGPUPointLight(
        int index, LightSource* lightSource, size_t frameNum, const osg::RefMatrix* viewMatrix)
    {
        auto* light = lightSource->getLight(frameNum);
        auto& buf = getUBOManager()->getLightBuffer(frameNum);
        buf->setDiffuse(index, light->getDiffuse());
        buf->setAmbient(index, light->getAmbient());
        buf->setSpecular(index, light->getSpecular());
        buf->setAttenuationRadius(index,
            osg::Vec4(light->getConstantAttenuation(), light->getLinearAttenuation(), light->getQuadraticAttenuation(),
                lightSource->getRadius()));
        buf->setPosition(index, light->getPosition() * (*viewMatrix));
    }

    osg::ref_ptr<osg::Uniform> LightManager::generateLightBufferUniform(const osg::Matrixf& sun)
    {
        osg::ref_ptr<osg::Uniform> uniform = new osg::Uniform(osg::Uniform::FLOAT_MAT4, "LightBuffer", getMaxLights());
        uniform->setElement(0, sun);

        return uniform;
    }

    void LightManager::setCollectPPLights(bool enabled)
    {
        if (enabled)
            mPPLightBuffer = std::make_shared<PPLightBuffer>();
        else
            mPPLightBuffer = nullptr;
    }

    LightSource::LightSource()
        : mRadius(0.f)
        , mActorFade(1.f)
        , mLastAppliedFrame(0)
    {
        setUpdateCallback(new CollectLightCallback);
        mId = sLightId++;
    }

    LightSource::LightSource(const LightSource& copy, const osg::CopyOp& copyop)
        : osg::Node(copy, copyop)
        , mRadius(copy.mRadius)
        , mActorFade(copy.mActorFade)
        , mLastAppliedFrame(copy.mLastAppliedFrame)
    {
        mId = sLightId++;

        for (size_t i = 0; i < mLight.size(); ++i)
            mLight[i] = new osg::Light(*copy.mLight[i].get(), copyop);
    }

    void LightListCallback::operator()(osg::Node* node, osgUtil::CullVisitor* cv)
    {
        bool pushedState = pushLightState(node, cv);
        traverse(node, cv);
        if (pushedState)
            cv->popStateSet();
    }

    bool LightListCallback::pushLightState(osg::Node* node, osgUtil::CullVisitor* cv)
    {
        if (!mLightManager)
        {
            mLightManager = findLightManager(cv->getNodePath());
            if (!mLightManager)
                return false;
        }

        if (!(cv->getTraversalMask() & mLightManager->getLightingMask()))
            return false;

        return false;

        // Possible optimizations:
        // - organize lights in a quad tree

        // Don't use Camera::getViewMatrix, that one might be relative to another camera!
        const osg::RefMatrix* viewMatrix = cv->getCurrentRenderStage()->getInitialViewMatrix();

        // Update light list if necessary
        // This makes sure we don't update it more than once per frame when rendering with multiple cameras
        if (mLastFrameNumber != cv->getTraversalNumber())
        {
            mLastFrameNumber = cv->getTraversalNumber();

            // Get the node bounds in view space
            // NB: do not node->getBound() * modelView, that would apply the node's transformation twice
            osg::BoundingSphere nodeBound;
            const osg::Transform* transform = node->asTransform();
            if (transform)
            {
                for (unsigned int i = 0; i < transform->getNumChildren(); ++i)
                    nodeBound.expandBy(transform->getChild(i)->getBound());
            }
            else
                nodeBound = node->getBound();

            transformBoundingSphere(*cv->getModelViewMatrix(), nodeBound);

            const std::vector<LightManager::LightSourceViewBound>& lights
                = mLightManager->getLightsInViewSpace(cv, viewMatrix, mLastFrameNumber);

            mLightList.clear();
            for (const LightManager::LightSourceViewBound& light : lights)
            {
                if (mIgnoredLightSources.contains(light.mLightSource))
                    continue;

                if (light.mViewBound.intersects(nodeBound))
                    mLightList.push_back(&light);
            }

            const size_t maxLights = mLightManager->getMaxLights() - mLightManager->getStartLight();

            if (mLightList.size() > maxLights)
            {
                // Sort by proximity to object: prefer closer lights with larger radius
                std::sort(mLightList.begin(), mLightList.end(),
                    [&](const SceneUtil::LightManager::LightSourceViewBound* left,
                        const SceneUtil::LightManager::LightSourceViewBound* right) {
                        const float leftDist = (nodeBound.center() - left->mViewBound.center()).length2();
                        const float rightDist = (nodeBound.center() - right->mViewBound.center()).length2();
                        // A tricky way to compare normalized distance. This avoids division by near zero
                        return left->mViewBound.radius() * rightDist > right->mViewBound.radius() * leftDist;
                    });

                mLightList.resize(maxLights);
            }
        }

        if (!mLightList.empty())
        {
            cv->pushStateSet(mLightManager->getLightListStateSet(mLightList, mLastFrameNumber, viewMatrix));
            return true;
        }
        return false;
    }

}

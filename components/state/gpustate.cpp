#include "gpustate.hpp"

#include <chrono>
#include <iomanip>

#include <osg/State>

#include <components/state/resourcemanager.hpp>

#include <components/debug/debuglog.hpp>

#include "texture.hpp"

namespace State
{
    GPUState::GPUState(std::shared_ptr<ResourceManager> resourceManager)
        : osg::Drawable()
        , mResourceManager(std::move(resourceManager))
    {
        getOrCreateStateSet()->setRenderBinDetails(-1, "RenderBin");
        setCullingActive(false);
        setUpdateCallback(new osg::NodeCallback);
    }

    void GPUState::makeGPUHandle(osg::State& state, BindlessTextureHandle& handle)
    {
        // Needs to map to the contextId, textures are resident per context
        if (handle.handle.has_value())
            return;

        ++mUploadCount;

        const unsigned int contextId = state.getContextID();

        auto* ext = state.get<osg::GLExtensions>();
        handle.texture->apply(state);
        state.applyTextureAttribute(0, handle.texture);

        osg::Texture::TextureObject* textureObject = handle.texture->getTextureObject(contextId);
        if (!textureObject)
            return;

        handle.handle = ext->glGetTextureHandle(textureObject->id());
    }

    int GPUState::open(osg::State& state, BindlessTextureHandle& handle)
    {
        // Needs to map to the contextId, textures are resident per context
        if (handle.resident)
        {
            mTextures[handle.virtualIndex] = *handle.handle;
            return handle.virtualIndex;
        }

        // Ensure the handle exists
        makeGPUHandle(state, handle);

        auto* ext = state.get<osg::GLExtensions>();
        ext->glMakeTextureHandleResident(*handle.handle);
        handle.resident = true;
        mTextures[handle.virtualIndex] = *handle.handle;

        // Log(Debug::Error) << "[RESIDENT] " << handle.handle.value() << " @VINDEX => " << handle.virtualIndex;
        return handle.virtualIndex;
    }

    void GPUState::close(osg::State& state, BindlessTextureHandle& handle)
    {
        if (!handle.resident)
            return;

        auto* ext = state.get<osg::GLExtensions>();
        ext->glMakeTextureHandleNonResident(*handle.handle);
        handle.resident = false;
    }

    void GPUState::drawImplementation(osg::RenderInfo& renderInfo) const
    {
        auto drawStart = std::chrono::high_resolution_clock::now();

        osg::State& state = *renderInfo.getState();
        const osg::GLExtensions* ext = state.get<osg::GLExtensions>();
        const unsigned int frameId = state.getFrameStamp()->getFrameNumber() % 2;

        auto& materials = mQueuedGPUMaterials[frameId];

        static bool initBuffers = true;
        if (initBuffers)
        {
            initBuffers = false;

            ext->glGenBuffers(1, &mSSBOMaterials);
            ext->glGenBuffers(1, &mSSBOTextures);
        }

        if (!materials.empty())
        {
            mMaterials.resize(std::max(mMaterials.size(), materials.rbegin()->first + 1));

            for (const auto& [index, material] : materials)
            {
                mMaterials[index] = material;
            }

            ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOMaterials);

            size_t newSize = sizeof(Material::GPUData) * mMaterials.size();

            // If the buffer hasn't been initialized or size has changed, update it
            if (mBufferSize == 0 || newSize > mBufferSize)
            {
                // Allocate new buffer size if the vector has grown or if it's uninitialized
                ext->glBufferData(GL_SHADER_STORAGE_BUFFER, newSize, nullptr, GL_DYNAMIC_DRAW);
                mBufferSize = newSize;
            }

            ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, newSize, mMaterials.data());

            ext->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            materials.clear();
        }

        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mSSBOMaterials);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Merge this frame's queued textures into the persistent pending queue
        auto& textures = mQueuedGPUTextures[frameId];
        if (!textures.empty())
        {
            mPendingTextures.insert(std::make_move_iterator(textures.begin()),
                std::make_move_iterator(textures.end()));
            textures.clear();
        }

        mUploadCount = 0;

        if (!mPendingTextures.empty())
        {
            mTextures.resize(std::max(mTextures.size(), mPendingTextures.rbegin()->first + 1));

            constexpr size_t maxUploadsPerFrame = 8;
            size_t uploads = 0;

            for (auto it = mPendingTextures.begin(); it != mPendingTextures.end(); )
            {
                auto& [index, handle] = *it;
                bool needsUpload = !handle.handle.has_value();

                if (needsUpload && uploads >= maxUploadsPerFrame)
                {
                    ++it;
                    continue;
                }

                if (handle.filename.empty())
                    Log(Debug::Error) << "Empty filename, this texture doesn't look right " << index;

                const_cast<GPUState*>(this)->open(state, handle);

                if (needsUpload)
                    ++uploads;

                it = mPendingTextures.erase(it);
            }

            ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOTextures);

            size_t newSize = sizeof(std::uint64_t) * mTextures.size();

            if (mTextureBufferSize == 0 || newSize > mTextureBufferSize)
            {
                ext->glBufferData(GL_SHADER_STORAGE_BUFFER, newSize, nullptr, GL_DYNAMIC_DRAW);
                mTextureBufferSize = newSize;
            }

            ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, newSize, mTextures.data());

            ext->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mSSBOTextures);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        auto drawEnd = std::chrono::high_resolution_clock::now();
        double drawMs = std::chrono::duration<double, std::milli>(drawEnd - drawStart).count();

        if (mUploadCount > 0 || drawMs > 5.0)
            Log(Debug::Error) << "[GPUState] frame " << state.getFrameStamp()->getFrameNumber()
                              << " uploads=" << mUploadCount
                              << " pending=" << mPendingTextures.size()
                              << " drawImpl=" << std::fixed << std::setprecision(1) << drawMs << "ms";
    }

    void GPUState::releaseGLObjects(osg::State* state) const {}

    void GPUState::traverse(osg::NodeVisitor& nv)
    {
        if (nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR)
        {
            size_t frameId = nv.getFrameStamp()->getFrameNumber() % 2;
            auto drained = mResourceManager->drainQueues();
            mQueuedGPUMaterials[frameId] = std::move(drained.materials);
            mQueuedGPUTextures[frameId] = std::move(drained.textures);
        }
    }
}